# 从单卡到多卡：分布式训练原理与 FSDP 实战

> 本文以 CodeChat 项目为例，解释为什么 8B 模型必须多卡训练，以及 FSDP 是怎么把一个放不下的模型"切"到 8 张卡上的。

---

## 1. 单卡训练：一切从这里开始

单卡训练最简单。一张 GPU 上放着三样东西：

```
┌─────────────────── GPU (80GB) ───────────────────┐
│  模型参数 (params)                                 │
│  梯度 (gradients)                                  │
│  优化器状态 (optimizer states)                      │
│  激活值 (activations，前向传播的中间结果)             │
└──────────────────────────────────────────────────┘
```

### 显存都花在哪了？

以 8B 参数模型为例（8.3B params）：

| 组件 | 计算方式 | 显存 |
|------|---------|------|
| 参数 (bf16) | 8.3B × 2 bytes | **~16 GB** |
| 梯度 (bf16) | 8.3B × 2 bytes | **~16 GB** |
| AdamW 一阶动量 m (fp32) | 8.3B × 4 bytes | **~32 GB** |
| AdamW 二阶动量 v (fp32) | 8.3B × 4 bytes | **~32 GB** |
| **合计（不含激活）** | | **~96 GB** |

一张 A800 只有 80GB，光是模型参数 + 优化器状态就需要 96GB，还没算激活值。

**这就是为什么 CodeChat 2B 能单卡训练，而 8B 不行。**

### CodeChat 2B 单卡的显存预算

```python
# scripts/base_train.py（单卡模式）
# 2B 模型：
#   参数 bf16:    ~5 GB
#   梯度 bf16:    ~5 GB
#   Adam m+v fp32: ~20 GB
#   激活（开启 grad_checkpoint）: ~10-20 GB
#   总计: ~40-50 GB  ← 80GB 卡放得下
```

---

## 2. 为什么不能简单地"多卡并行"？

### 2.1 数据并行 (Data Parallelism / DDP)

最朴素的多卡方案：每张卡上放一份**完整的模型**，各自处理不同的数据，然后同步梯度。

```
DDP: 每张卡都有完整的一份

GPU 0: [全部参数] + [全部梯度] + [全部优化器状态]  ← 还是 96GB
GPU 1: [全部参数] + [全部梯度] + [全部优化器状态]  ← 还是 96GB
...
GPU 7: [全部参数] + [全部梯度] + [全部优化器状态]  ← 还是 96GB
```

**DDP 解决了速度问题（8 张卡处理 8 倍数据），但没有解决显存问题。** 每张卡仍然需要放下完整的 96GB，单卡 80GB 还是不够。

PyTorch 中 DDP 的核心操作是 **AllReduce**——所有卡在反向传播后同步梯度的平均值：

```
反向传播后:
  GPU 0 梯度: [g0_0, g0_1, g0_2, ...]
  GPU 1 梯度: [g1_0, g1_1, g1_2, ...]
  ...

AllReduce 后 (每张卡都拿到平均梯度):
  GPU 0 梯度: [avg_0, avg_1, avg_2, ...]
  GPU 1 梯度: [avg_0, avg_1, avg_2, ...]   ← 全部相同
  ...
```

### 2.2 朴素模型并行 (Naive Model Parallelism)

把模型按层切开，不同层放在不同卡上：

```
GPU 0: Layer 0-9    →  前向传播  →  传给 GPU 1
GPU 1: Layer 10-19  →  前向传播  →  传给 GPU 2
GPU 2: Layer 20-29  →  前向传播  →  传给 GPU 3
GPU 3: Layer 30-39  →  前向传播  →  输出
```

**问题：流水线气泡 (pipeline bubble)**。当 GPU 3 在算前向时，GPU 0/1/2 全在等着，利用率极低。

### 2.3 真正的需求

我们需要一种方案，**既切分显存（解决放不下的问题），又让所有卡都在干活（保持高利用率）**。

这就是 FSDP 要解决的问题。

---

## 3. FSDP：把参数、梯度、优化器全部切碎

**FSDP = Fully Sharded Data Parallel**，是 PyTorch 原生的零冗余并行方案，等价于 DeepSpeed ZeRO-3。

核心思想：**不再让每张卡持有完整模型，而是把参数/梯度/优化器状态均匀切片(shard)到所有卡上。需要用的时候临时聚合(all-gather)，用完立刻丢弃。**

### 3.1 三个层级的切分

微软 DeepSpeed 论文把零冗余优化分成三个阶段：

| 阶段 | 切什么 | 每卡显存 (8B, 8 卡) | 等价 |
|------|--------|---------------------|------|
| ZeRO-1 | 仅优化器状态 | ~24 GB (params+grads 仍完整) | — |
| ZeRO-2 | 优化器 + 梯度 | ~20 GB | FSDP `SHARD_GRAD_OP` |
| **ZeRO-3** | **优化器 + 梯度 + 参数** | **~12 GB** | **FSDP `FULL_SHARD`** |

CodeChat 使用 `FULL_SHARD`（ZeRO-3），三样全切：

```python
# scripts/base_train.py / chat_sft.py / chat_rl.py
ShardingStrategy.FULL_SHARD   # params + grads + optim 全部按 rank 切片
```

### 3.2 FSDP 的运行流程

以一个 forward → backward → optimizer step 为例：

```
═══════════════════════════════════════════════════════
  FORWARD PASS (前向传播)
═══════════════════════════════════════════════════════

平时每张卡只持有 1/8 的参数：

  GPU 0: [shard_0]     (参数的第 0 片，~2GB)
  GPU 1: [shard_1]     (参数的第 1 片，~2GB)
  ...
  GPU 7: [shard_7]     (参数的第 7 片，~2GB)

当需要计算某一层时，FSDP 临时执行 All-Gather：

  ┌─── All-Gather ────────────────────────────────┐
  │ GPU 0 广播 shard_0  ──→  所有卡               │
  │ GPU 1 广播 shard_1  ──→  所有卡               │
  │ ...                                            │
  │ GPU 7 广播 shard_7  ──→  所有卡               │
  │                                                │
  │ 结果: 每张卡临时持有这一层的完整参数             │
  └────────────────────────────────────────────────┘

  → 用完整参数做前向计算
  → 计算完毕后立即丢弃非本 rank 的参数 (释放显存)
  → 进入下一层，重复 All-Gather

═══════════════════════════════════════════════════════
  BACKWARD PASS (反向传播)
═══════════════════════════════════════════════════════

  → 再次 All-Gather 该层参数 (前向时已丢弃)
  → 计算该层梯度
  → Reduce-Scatter: 每张卡只保留属于自己那片的梯度
  → 释放完整梯度，只留 1/8

  ┌─── Reduce-Scatter ────────────────────────────┐
  │ 把梯度切成 8 片，每张卡拿到自己那片的求和结果   │
  │                                                │
  │ GPU 0: grad_shard_0  (对应 param_shard_0)      │
  │ GPU 1: grad_shard_1  (对应 param_shard_1)      │
  │ ...                                            │
  └────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════
  OPTIMIZER STEP (优化器更新)
═══════════════════════════════════════════════════════

  每张卡只更新自己的 1/8:

  GPU 0: Adam(param_shard_0, grad_shard_0, m_shard_0, v_shard_0)
  GPU 1: Adam(param_shard_1, grad_shard_1, m_shard_1, v_shard_1)
  ...

  无需通信！每张卡独立更新自己的碎片。
```

### 3.3 显存对比

8B 模型，8×A800：

```
DDP (每卡完整副本):
  参数:    16 GB
  梯度:    16 GB
  Adam:    64 GB
  ─────────────
  总计:    96 GB  ← 超出 80GB，OOM！

FSDP FULL_SHARD (每卡 1/8):
  参数:     2 GB  (16/8)
  梯度:     2 GB  (16/8)
  Adam:     8 GB  (64/8)
  All-Gather 临时缓冲:  ~2-4 GB (一层的完整参数)
  激活值:   ~10-20 GB
  ─────────────
  总计:    ~24-36 GB  ← 80GB 轻松放下
```

---

## 4. FSDP 的通信机制

### 4.1 两种核心通信原语

```
All-Gather: 每张卡贡献自己的碎片，所有卡拿到完整结果
┌───┐ ┌───┐ ┌───┐ ┌───┐
│ A │ │ B │ │ C │ │ D │   (每卡一片)
└───┘ └───┘ └───┘ └───┘
          ↓ All-Gather
┌───────────────────────┐
│ A │ B │ C │ D │         (每卡都拿到全部)
└───────────────────────┘ × 4 份

Reduce-Scatter: 先求和再分发，每张卡拿到一片的求和结果
GPU 0: [a0, a1, a2, a3]
GPU 1: [b0, b1, b2, b3]
GPU 2: [c0, c1, c2, c3]
GPU 3: [d0, d1, d2, d3]
          ↓ Reduce-Scatter
GPU 0: [a0+b0+c0+d0]         (第 0 片的总和)
GPU 1: [a1+b1+c1+d1]         (第 1 片的总和)
GPU 2: [a2+b2+c2+d2]         (第 2 片的总和)
GPU 3: [a3+b3+c3+d3]         (第 3 片的总和)
```

### 4.2 NVLink 的作用

A800 之间通过 NVLink 互联，带宽 ~400 GB/s（双向），远高于 PCIe 的 ~32 GB/s。

FSDP 中 All-Gather 一层 8B 模型的参数量大约 400MB，在 NVLink 上只需 ~1ms。这使得"用时聚合、用完丢弃"的策略在实际中几乎不增加额外开销。

### 4.3 通信与计算重叠

FSDP 的一个关键优化是 **backward prefetch**：在计算第 N 层反向传播的同时，提前 All-Gather 第 N-1 层的参数。

```python
# CodeChat 中的配置
backward_prefetch=BackwardPrefetch.BACKWARD_PRE  # 提前预取
```

```
时间线:
  ┌── GPU 计算 ──┐┌── GPU 计算 ──┐
  │ Layer N 反向  ││ Layer N-1 反向│
  └──────────────┘└──────────────┘
  ┌── NVLink ────────┐
  │ All-Gather N-1   │  ← 与 Layer N 计算并行
  └──────────────────┘
```

---

## 5. FSDP 的切分粒度：Auto Wrap Policy

FSDP 不是把整个模型当作一块来切，而是按 **transformer block (层)** 为单位包装。每个 Block 是一个独立的 FSDP 单元。

```python
# CodeChat 的包装策略
auto_wrap = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={Block},   # ← 每个 Block 单独包一层 FSDP
)
```

为什么按 Block 切？

```
不切 (整个模型一个 FSDP 单元):
  All-Gather 整个模型 → 临时需要 16GB → 显存峰值太高

按 Block 切 (40 个 FSDP 单元):
  每次 All-Gather 一个 Block → 临时需要 ~400MB → 显存峰值可控
```

模型结构与 FSDP 包装的对应关系：

```
GPT (顶层 FSDP)
├── wte (embedding)           ← 跟顶层一起切
├── Block 0  (FSDP 单元 0)    ← 独立切片
│   ├── LayerNorm
│   ├── MultiHeadAttention
│   ├── LayerNorm
│   └── MLP
├── Block 1  (FSDP 单元 1)    ← 独立切片
│   └── ...
├── ...
├── Block 39 (FSDP 单元 39)   ← 独立切片
├── ln_f (final norm)          ← 跟顶层一起切
└── lm_head                    ← 跟顶层一起切
```

---

## 6. 混合精度训练 (Mixed Precision)

FSDP 支持三种粒度的精度控制：

```python
mp_policy = MixedPrecision(
    param_dtype=COMPUTE_DTYPE,    # bf16: 参数在 GPU 上以 bf16 存储和计算
    reduce_dtype=COMPUTE_DTYPE,   # bf16: 梯度同步时用 bf16 通信
    buffer_dtype=COMPUTE_DTYPE,   # bf16: BatchNorm 等 buffer 也用 bf16
)
```

| 组件 | 精度 | 原因 |
|------|------|------|
| 参数 (前向/反向) | bf16 | 节省显存和计算，A800 有 bf16 Tensor Core |
| 梯度通信 | bf16 | 减少 NVLink 传输量 |
| **AdamW 状态 (m, v)** | **fp32** | 优化器精度必须高，否则训练不稳定 |
| **优化器更新** | **fp32** | 小的梯度更新在 bf16 下会被截断为 0 |

这就是为什么 Adam 状态占最多显存——它必须用 fp32：

```
bf16 参数:      8.3B × 2 bytes = 16 GB
fp32 Adam m:    8.3B × 4 bytes = 32 GB  ← 是参数的 2 倍！
fp32 Adam v:    8.3B × 4 bytes = 32 GB
```

---

## 7. CodeChat 从单卡到多卡的实际改动

### 7.1 启动方式

```bash
# 单卡 (2B)
python -m scripts.base_train --preset 2b ...

# 多卡 (8B, 8×A800)
torchrun --nproc_per_node=8 -m scripts.base_train --preset 8b ...
```

`torchrun` 会启动 8 个进程，每个进程设置 `LOCAL_RANK=0..7` 环境变量。

### 7.2 分布式初始化

```python
def setup_distributed():
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 0, 1            # 单卡模式，什么都不做
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)     # 每个进程绑定自己的 GPU
    dist.init_process_group(backend="nccl")  # 初始化 NCCL 通信后端
    return True, local_rank, dist.get_rank(), dist.get_world_size()
```

### 7.3 Checkpoint 加载

```python
# 错误做法 (所有进程同时加载到 GPU 0):
state = load_ckpt(path)  # 默认 map_location="cuda" → 全部挤到 GPU 0

# 正确做法 (先加载到 CPU，再分发):
state = load_ckpt(path, map_location="cpu")  # CPU 内存充足
model.load_state_dict(state["model"])
del state  # 释放 CPU 内存
model = model.to(device)  # 移到各自的 GPU
```

### 7.4 FSDP 包装

```python
model = model.to(device).to(COMPUTE_DTYPE)
if is_dist:
    model = wrap_fsdp(model)  # 参数被切片到各卡

# 之后 model 的行为对外不变：
#   logits, loss = model(x, y)   ← 内部自动 All-Gather + Reduce-Scatter
```

### 7.5 梯度同步优化：no_sync

当使用梯度累积时，只有最后一个 micro-batch 需要同步梯度：

```python
for micro in range(grad_accum):
    x, y = loader.next_batch()
    if is_dist and micro < grad_accum - 1:
        with model.no_sync():          # ← 跳过中间步的梯度同步
            _, loss = model(x, y)
            (loss / grad_accum).backward()
    else:                              # ← 最后一步才真正同步
        _, loss = model(x, y)
        (loss / grad_accum).backward()
```

没有 `no_sync` 的话，每个 micro-batch 都会触发 Reduce-Scatter，白白浪费 7 次通信。

### 7.6 梯度裁剪

```python
# 单卡:
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# FSDP: 必须用 FSDP 自己的方法，因为参数是分片的
grad_norm = model.clip_grad_norm_(1.0)
```

FSDP 的 `clip_grad_norm_` 内部会先 All-Gather 梯度范数，计算全局范数后再裁剪。

### 7.7 Checkpoint 保存

```python
# codechat/checkpoint.py
if _is_fsdp(model):
    # FSDP 模型的参数分散在各卡，需要先聚合
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, ...):
        model_state = model.state_dict()  # 在 rank 0 上聚合完整参数
    # 只有 rank 0 写文件
```

### 7.8 只在 rank 0 做 I/O

```python
writer = SummaryWriter(log_dir=tb_path) if is_master else None

if is_master:
    print(f"step {step} | loss {loss:.4f}")
    writer.add_scalar("train/loss", loss, step)
```

如果 8 个进程都写 TensorBoard，日志会重复 8 份且可能互相覆盖。

---

## 8. RL 阶段的特殊处理

RL (GRPO) 比 SFT 更复杂，因为它需要**采样**——自回归生成 token。在 FSDP 下有两个额外的同步点：

### 8.1 问题索引同步

每个 step 随机选一个 MBPP 问题。8 个进程必须选同一个：

```python
idx = torch.randint(0, len(mbpp), (1,), device=device)
if is_dist:
    dist.broadcast(idx, src=0)  # rank 0 的选择广播给所有 rank
```

### 8.2 采样 token 同步

自回归采样时，`torch.multinomial` 在不同 GPU 上可能产生不同结果（CUDA RNG 差异）。所有 rank 必须生成相同的 token 序列，否则后续的 forward 输入不一致会导致 FSDP 死锁。

```python
nxt = torch.multinomial(probs, 1)     # 各 rank 可能不同
if is_dist:
    dist.broadcast(nxt, src=0)         # 强制所有 rank 用 rank 0 的结果
```

---

## 9. 各种并行策略对比

| 策略 | 切什么 | 适用场景 | 通信量 | CodeChat 使用 |
|------|--------|---------|--------|---------------|
| **DDP** | 不切，同步梯度 | 模型放得下单卡 | 低 (AllReduce 梯度) | 2B 可用 |
| **FSDP SHARD_GRAD_OP** (ZeRO-2) | 梯度 + 优化器 | 参数放得下但优化器放不下 | 中 | — |
| **FSDP FULL_SHARD** (ZeRO-3) | 参数 + 梯度 + 优化器 | 模型放不下单卡 | 高 (每层 All-Gather) | **8B 使用** |
| **Tensor Parallel** | 切矩阵乘法 | 超大模型 (70B+) | 每层内通信 | — |
| **Pipeline Parallel** | 切层到不同卡 | 超大模型 + 多节点 | 层间传递激活 | — |

实际中大模型训练常常**组合使用**多种策略（如 Llama 3 用了 FSDP + Tensor Parallel + Pipeline Parallel）。CodeChat 8B 只需 FSDP 就够了。

---

## 10. 总结

```
单卡 2B:   什么都不用想，model.to("cuda") 就行
            显存 ~40GB，一张 A800 足够

多卡 8B:   DDP 不够（每卡仍需 96GB）
            必须用 FSDP FULL_SHARD（每卡 ~24GB）
            代价：每层前向/反向各一次 All-Gather + Reduce-Scatter
            NVLink 高带宽 (~400 GB/s) 使通信开销可接受

代码改动:   5 处核心改动
            1. setup_distributed() — 初始化 NCCL
            2. map_location="cpu" — checkpoint 先载 CPU
            3. wrap_fsdp(model) — FSDP 包装
            4. model.no_sync() — 梯度累积优化
            5. model.clip_grad_norm_() — FSDP 感知的梯度裁剪
```

FSDP 的精髓就一句话：**用通信换显存，让 N 张卡合力装下一个本来放不下的模型，同时保持接近 N 倍的训练速度。**
