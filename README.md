# CodeChat 8B — 8x A800 训练说明

本 README 专门介绍在 **8x NVIDIA A800-SXM4-80GB** 上把 CodeChat 从 **2B 升级到 8B** 的这次训练：动机、FSDP 分片方案、显存预算、启动命令、代码改动、续训策略、常见坑。与主 [`README_A800_x1.md`](README_A800_x1.md) 的单卡 2B 路线互补，不替代。

---

## TL;DR

```bash
# 一键 (data + 8B pretrain + SFT + RL)
# 首次运行会自动创建 .venv_train（复用系统 torch），再装 tensorboard 等缺的包
bash runs/train_a800_x8.sh

# Function-calling 能力续 SFT (从 8B_sft checkpoint 继续，glaiveai/glaive-function-calling-v2)
bash runs/train_a800_x8_v2_funcall.sh

# 同一条 pipeline 带上"救活 MBPP RL"3 阶段 (pass@k 诊断 → 难度过滤 → 优化版 GRPO)
RUN_RL=1 bash runs/train_a800_x8_v2_funcall.sh

# 或者只跑预训练 (venv 已建好后)
./.venv_train/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node=8 \
    -m scripts.base_train \
    --data-dir data/pretrain \
    --preset 8b \
    --device-batch-size 1 \
    --grad-accum 8 \
    --max-steps 30000 \
    --lr 1.5e-4 --warmup 1000 \
    --run-name codechat_8b
```

> ⚠️ **一定用 `--run-name` 而不是 `--run`**：`torchrun` 的 argparse 有个 `--run-path` 选项，看到 `--run` 会判为 "ambiguous option" 然后直接报错（在还没把参数转发给训练脚本之前就挂了）。`--run-name` 不是它任何选项的前缀，才会被透传。base_train/chat_sft/chat_rl 里同时兼容 `--run` (单卡老脚本) 和 `--run-name` (torchrun 场景)。

## Python 环境 (`.venv_train`)

这台 8x A800 的宿主上 root 不能往系统 site-packages 里安装东西，因此不能直接 `pip install -r requirements.txt`。`runs/train_a800_x8.sh` 首次运行时会自动做这几件事：

1. **建一个 venv**：`python3 -m venv --system-site-packages .venv_train`
   - `--system-site-packages` 让 venv 继承系统安装的 torch / CUDA / nvidia-* wheel，**不重装 torch** (重装 2.10.0+cu130 会非常慢且可能装坏)。
2. **pip 装训练缺的包到 venv 里**：tensorboard、tiktoken、numpy、tqdm、datasets(<4.0)、huggingface_hub(<0.24)、fsspec、pyarrow、requests、httpx[socks]。这些安装在 `.venv_train/lib/python3.12/site-packages/`，**不污染系统**。
3. **sanity check**：venv 的 python 能 import 到系统 torch，并打印 `torch.__version__ / cuda / n_gpu`。
4. **用 `python -m torch.distributed.run` 代替 `torchrun`**：venv 的 `bin/` 里没有 torchrun 的入口脚本（因为 torch 来自系统 site-packages），所以直接用模块形式启动，效果完全等价。
5. **设置 `PYTHONPATH=$REPO_ROOT`**：项目没 `pip install -e .`，靠 PYTHONPATH 让 `import codechat.*` 能找到仓库目录。

宿主环境参考：

| 项目 | 版本 |
|---|---|
| OS | Ubuntu (/mnt/openclaw/CodeChat) |
| Python | 3.12.3 |
| torch | 2.10.0+cu130 (系统预装) |
| CUDA | 13.0 |
| GPU | 8x A800-SXM4-80GB |

如果宿主 python 不叫 `python3`，用 `SYS_PYTHON=/path/to/python bash runs/train_a800_x8.sh` 覆盖。

**常用操作**：

```bash
# 看 venv 里装了什么
./.venv_train/bin/pip list

# 手动进入 venv (交互 debug 用)
source .venv_train/bin/activate

# 想重建 venv: 直接删
rm -rf .venv_train && bash runs/train_a800_x8.sh
```

- 预设 `8b`:  `depth=40, n_embd=4096, n_head=32`, **~8.3B 参数**
- 分片策略: **FSDP `FULL_SHARD`** (params + grads + AdamW 状态全部按 rank 切 8 份)
- 计算精度: bf16 (A800 原生支持)
- 全局 batch: `1 × 8 × 8 × 2048 ≈ 131k tokens/step`

---

## 为什么从 2B 升到 8B 要换方案

| 资源项 | 2B (单 A800) | 8B (单 A800) | 8B (8x A800 FSDP) |
|---|---|---|---|
| 权重 bf16 | 4.2 GB | 16 GB | 16/8 = 2 GB |
| 梯度 bf16 | 4.2 GB | 16 GB | 16/8 = 2 GB |
| AdamW m+v (fp32) | 17 GB | **64 GB** | 64/8 = 8 GB |
| 参数 fp32 主副本 | 8.5 GB | **32 GB** | 32/8 = 4 GB |
| 激活 (bs=1, seq=2048, grad_ckpt) | ~25 GB | ~35 GB | ~35 GB |
| **峰值** | ~60 GB ✅ | **~160 GB ❌** | **~55 GB ✅** |

一句话：**单张 80GB 卡放不下 8B 的优化器状态**，光 fp32 的 AdamW m+v 就是 64GB。必须把 params / grads / optim 三样东西切片到多卡，**FSDP `FULL_SHARD` 正是干这件事的**（等价于 DeepSpeed ZeRO-3）。DDP 不行，因为 DDP 只同步梯度，每卡仍然保留一份完整的 params + optim。

**关键决策: FSDP 而非 DDP / ZeRO-2 / Tensor Parallel**

- DDP：每卡各有完整优化器状态 → 单卡仍要 ~160GB，排除
- ZeRO-2：只切 grads + optim，params 仍然整份 → 每卡仍要 ~50GB params+acts+grads，紧张
- FSDP `FULL_SHARD` (= ZeRO-3)：三样都切 → 每卡 ~55GB ✅
- Tensor Parallel：需要改模型代码（拆 attention / MLP 的权重），工作量大，8 卡单机收益不如 FSDP 明显
- Pipeline Parallel：对 depth=40 可以做，但要搬 micro-batch 调度，代码改动量远高于 FSDP

所以 **FSDP + 激活检查点 + bf16** 是单机 8 卡 80GB 跑 8B 最小改动的组合，这也是本仓库采用的方案。

---

## 模型规格 (preset=`8b`)

| 参数 | 2B (旧) | **8B (新)** |
|---|---|---|
| `depth` | 32 | **40** |
| `n_embd` | 2560 | **4096** |
| `n_head` | 20 | **32** (head_dim=128) |
| `block_size` | 2048 | 2048 |
| 参数量 | ~2.1B | **~8.3B** |
| 词表 | 50257 (GPT-2 BPE, tied) | 同左 |
| 激活检查点 | 开 | 开 (仍然必须) |

8B 的公式估算：`12·L·d² + V·d = 12·40·4096² + 50257·4096 ≈ 8.05B + 0.21B ≈ 8.26B`，与实际 `numel()` 一致。

---

## FSDP 具体怎么配的

见 `scripts/base_train.py:40-64` 的 `wrap_fsdp()`：

```python
FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,     # ZeRO-3
    mixed_precision=MixedPrecision(
        param_dtype=bf16, reduce_dtype=bf16, buffer_dtype=bf16),
    auto_wrap_policy=transformer_auto_wrap_policy(
        transformer_layer_cls={Block}),                 # 每个 Block 一个 shard 单元
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,    # 反向时预取下一层 params
    device_id=torch.cuda.current_device(),
    use_orig_params=True,                               # 让 optimizer/clip 看到原 shape
    limit_all_gathers=True,                             # 控制 all-gather 并发，避免打爆 NCCL
)
```

几个非默认选项的理由：

- **`transformer_auto_wrap_policy(Block)`**: 按 `codechat.gpt.Block` 切片，40 层 = 40 个 shard 单元，粒度合适（太粗显存峰值高，太细 NCCL 开销大）。
- **`use_orig_params=True`**: 让 `optim.param_groups` / `clip_grad_norm_` 看到原始形状的参数，`build_optimizer` 里 `p.ndim >= 2` 的 decay/no_decay 分组才能继续工作，无需改 `codechat/optim.py`。
- **`backward_prefetch=BACKWARD_PRE`**: 反向传播时提前 all-gather 下一层 params，用通信-计算重叠把吞吐再挤一点。
- **`model.no_sync()`** 在非最后一次 micro-step 上：`grad_accum=8` 时只在第 8 次真正同步梯度，省掉 7 次 all-reduce。

---

## 启动命令 & 调参

### 一键流水线

```bash
bash runs/train_a800_x8.sh
```

脚本分 5 个阶段，用 `SKIP_TO=N` 可以从第 N 段起续：

| Stage | 内容 | 说明 |
|---|---|---|
| 1 | 预训练数据准备 | 已有 `data/pretrain/*.bin` 可注释掉 |
| 2 | **8B 预训练 (FSDP x8)** | 30k step ✅ DONE (~48.9h) |
| 3 | SFT 数据准备 | |
| 4 | 8B SFT (FSDP x8) | `chat_sft.py` 已补 FSDP 支持 ✅ DONE (~4.8h) |
| 5 | RL (GRPO on MBPP) | `chat_rl.py` 已补 FSDP；但 base 能力不足，reward 恒 0 ⚠️ 详见 "MBPP RL 救活指南" |

### 关键超参 (base_train)

| 参数 | 默认 (单卡 2B) | **8x A800 / 8B** | 说明 |
|---|---|---|---|
| `--preset` | `2b` | **`8b`** | 模型大小 |
| `--device-batch-size` | 2 | **1** | 8B 每卡放不下 2，降到 1 |
| `--grad-accum` | 16 | **8** | world_size 已经 ×8，总 batch 保持量级 |
| `--lr` | 2e-4 | **1.5e-4** | 大模型略降 lr |
| `--warmup` | 500 | **1000** | 大模型多 warmup 更稳 |
| `--max-steps` | 30000 | 30000 | 同 |
| `--save-every` | 2000 | **1000** | 8B 贵，多留快照 |

**全局 batch 计算**: `device_bs × grad_accum × world_size × block_size = 1 × 8 × 8 × 2048 ≈ 131k tokens/step`。30k step ≈ 3.9B tokens，约等于 Chinchilla 对 8B 建议量的 ~2.4%（仅用作代码 domain 的轻量训练，非充分预训练）。

---

## 这次为支持 8B / FSDP 改了哪些代码

| 文件 | 改动 | 目的 |
|---|---|---|
| `codechat/gpt.py` | `PRESETS` 增加 `"8b"` | depth=40, n_embd=4096, n_head=32 |
| `codechat/dataloader.py` | `PretrainLoader` 增加 `rank` / `world_size` / 独立 `np.random.default_rng(seed + rank*9973)` | 不同 rank 流到不同 random window，避免 8 卡看同一批数据 |
| `codechat/checkpoint.py` | `save()` 走 `FSDP.state_dict_type(FULL_STATE_DICT, offload_to_cpu=True, rank0_only=True)` 把切片 gather 回 rank 0 | FSDP 下存一份和单卡兼容的完整权重，下游 `chat_cli` / `eval_swebench` 无需知道训练是不是 FSDP |
| `scripts/base_train.py` | 自动检测 `LOCAL_RANK`，用 `torchrun` 拉起时走 FSDP 分支；增加 `model.no_sync()`、FSDP `clip_grad_norm_`、rank-0-only 的 TB/log；新增 `--resume` 只加载 weights | 单 API，单卡和 8 卡共用一个脚本 |
| `runs/train_a800_x8.sh` | **新文件**，5 段 `torchrun --nproc_per_node=8 -m scripts.base_train ...` | 8 卡一键入口 |

**向后兼容**：单卡 `python -m scripts.base_train --preset 2b ...` 行为完全不变，检测不到 `LOCAL_RANK` 就走老路径。老的 `runs/train_a800.sh` 也无需动。

---

## 显存 / 吞吐预期

（单机 8x A800-SXM4-80GB + NVLink/NVSwitch，bf16，block_size=2048）

| 指标 | 预期 |
|---|---|
| 每卡峰值显存 | ~55–65 GB |
| 单步 wall time | ~6–10 s |
| 吞吐 (global) | ~15–25 Ktok/s |
| 30k step 总耗时 | **~50–80 h** |

数字偏估算 —— 实测可能随 NVSwitch 拓扑、NCCL 版本、激活检查点粒度上下浮动 20%。第一次跑建议先用 `--max-steps 200` 短跑一遍，看 TB 里的 `perf/ktok_per_s` 再决定要不要调 `grad-accum` / `device-batch-size`。

---

## 续训 & 从 2B 的 checkpoint 出发？

**不能直接续训**。2B (`depth=32, n_embd=2560`) 和 8B (`depth=40, n_embd=4096`) 维度不兼容，`model.load_state_dict(strict=True)` 会直接报错；即便 `strict=False`，hidden 维度不匹配的层也无法复用。

有这么几个选项：

1. **从零开始 (本脚本默认)**：最干净。8B 用新的 `run` 名 `codechat_8b`，和 `codechat_2b` 互不干扰，你之前 step 22290 的 2B 权重、TensorBoard 曲线都还在。
2. **Depth up-scaling (推荐的"半续训")**：把 2B 的 32 层权重复制进 8B 的前 32 层，后 8 层随机初始化或复制倒数几层。需要写一段手动 loader，本仓库未内置。
3. **知识蒸馏**：把 2B 当 teacher，给 8B 训练时加 KL loss。改动量最大，暂不考虑。

当前 `runs/train_a800_x8.sh` 按选项 1 写。需要 2 或 3 时告诉我，我再加 `--init-from-ckpt` 这类开关。

---

## 保存下来的 checkpoint 长什么样

FSDP 训练时每隔 `--save-every` 步触发一次 `save_ckpt`：

1. `FSDP.state_dict_type(FULL_STATE_DICT, rank0_only=True, offload_to_cpu=True)` 上下文里 gather
2. 非 rank-0 直接 `return`，rank-0 落盘到 `checkpoints/codechat_8b/latest.pt`
3. 所有 rank 在 `dist.barrier()` 上对齐

结果：**盘上看到的是一份完整 `state_dict`**（和单卡训练存下来的 2B 长得一样），可以用 `chat_cli` / `eval_swebench` 直接加载，不需要 FSDP 环境。代价是保存瞬间 rank 0 的 CPU 内存要能放下 8B fp32 ≈ 32GB，一般 A800 主机 RAM 都够。

**没存 optimizer state**：FSDP 的 optimizer state sharded-save 要另外走 `FSDP.optim_state_dict_to_save`，比较繁琐。预训练里丢失 optim 只会让 momentum 冷启动，影响很小；需要完整 resume 时再加。

---

## 常见坑 / 检查清单

- **NCCL 卡住 / 超时**：先确认 `NCCL_DEBUG=INFO` 下 8 张卡都列出，NVLink/NVSwitch 有效。单机单节点用 `--standalone` 即可，不需要 `--rdzv-endpoint`。
- **`RuntimeError: only FullShardedDataParallel supports …`**：检查 `use_orig_params=True` 是否传了，没传的话 `build_optimizer` 里 `p.ndim` 会拿到 flat 1D 形状导致 decay 分组失效。
- **OOM 在第 1 步**：通常是 **NCCL buffer + 激活 peak** 撞上。先把 `--device-batch-size` 固定 1，再把 `block_size` 调到 1024 临时验证，最后调回 2048。
- **OOM 在 save 时**：rank 0 CPU 内存不够装 8B fp32。确认主机 RAM ≥ 64GB；否则把 `offload_to_cpu=False` 改成直接从 GPU 盘存（代价是 rank-0 GPU 要装一份完整 ~32GB，80GB 卡够）。
- **速度远低于预期 (<5 Ktok/s)**：先看 `nvidia-smi topo -m` 是不是所有卡都走 NVLink；其次看 `grad_accum=8` 里 7 次 `no_sync` 是否生效（FSDP 下 `no_sync` 的收益比 DDP 小但仍有 ~10-20%）。
- ~~**chat_sft.py / chat_rl.py 当前不支持 FSDP**~~（已解决）：`scripts/chat_sft.py` 和 `scripts/chat_rl.py` 均已按 `base_train.py` 的 pattern 加好 `setup_distributed + wrap_fsdp`（commits `bc8c26b` / `8bcc723`）。`torchrun --nproc_per_node=8 -m scripts.chat_sft|chat_rl` 直接可用，单卡调用仍走老路径。

---

## 对比：`train_a800*.sh` 家族

| 脚本 | 目标硬件 | 模型 | 分布式 | 适合场景 |
|---|---|---|---|---|
| `train_a800.sh` | 1x A800 | 2B | 无 | 主 pipeline 的入门路线（主 README 描述） |
| `train_a800_v2.sh` | 1x A800 | 2B | 无 | 在 v1 基础上追加 SWE-bench eval + docker RL |
| **`train_a800_x8.sh`** | **8x A800** | **8B** | **FSDP FULL_SHARD** | **本文主角**：放大到 8B，拿满 8 卡吞吐 |
| **`train_a800_x8_v2_funcall.sh`** | **8x A800** | **8B** | **FSDP** | funcall 续 SFT + "救活 MBPP RL" 的优化版 3 阶段（见下） |

四个脚本互不替代，按手里机器和目标选。

---

## Function-calling 能力训练 (`train_a800_x8_v2_funcall.sh`)

从 `checkpoints/codechat_8b_sft/latest.pt` 基础上，用 [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)（~113k 多轮对话）继续 SFT，让模型学会：

- 看到 SYSTEM 里声明的函数 + USER 请求后，输出 `<functioncall> {"name": ..., "arguments": ...}` 的 JSON 调用
- 收到 FUNCTION RESPONSE 后用自然语言总结给用户

**聊天格式**（复用 GPT-2 BPE，不改 tokenizer，新 role tag 直接 BPE）：

```
<|system|>\n{function schema}\n<|end|>\n
<|user|>\n{query}\n<|end|>\n
<|assistant|>\n<functioncall> {...json...}\n<|end|>\n
<|function_response|>\n{tool output}\n<|end|>\n
<|assistant|>\n{final answer}\n<|end|>\n
```

loss 只计算 assistant 段的 token（system / user / function_response 全部 `-100` 掩掉），让模型专门学"该在什么时候发 functioncall"和"响应回来怎么措辞"。

**最小运行**：

```bash
# 默认只跑 funcall SFT (stage 1-2)
bash runs/train_a800_x8_v2_funcall.sh

# 快速 smoke test (2000 条样本)
MAX_EXAMPLES=2000 bash runs/train_a800_x8_v2_funcall.sh

# 完整跑：funcall SFT + 优化版 MBPP RL (stage 1-5)
RUN_RL=1 bash runs/train_a800_x8_v2_funcall.sh
```

**文件**：

| 文件 | 作用 |
|---|---|
| `scripts/prepare_sft_funcall.py` | 解析 glaive 的 `SYSTEM:/USER:/A:/FUNCTION RESPONSE:` 标记，产出 `{input_ids, labels}` jsonl |
| `runs/train_a800_x8_v2_funcall.sh` | 5-stage pipeline (SFT + 可选 RL) |

---

## MBPP RL 救活指南（`RUN_RL=1` 的 3 阶段）

首轮 `train_a800_x8.sh` 的 stage 5 RL 跑了 415 步 ≈ 16.7h，**reward 恒等于 0.000**（详见 [`reports/TRAINING_REPORT_8b_a88_x8.md`](reports/TRAINING_REPORT_8b_a88_x8.md#5-阶段-5-rl--grpo--已跑通但无收益-reward0)）。根因：group_size=4 的 4 个 rollout 一次都没过 MBPP test → advantage 全 0 → GRPO 无梯度信号，训练无意义。

`train_a800_x8_v2_funcall.sh` 里新增的 stage 3-5 是对应的优化方案，对齐训练报告 §5.6 的性价比清单：

| Stage | 脚本 | 优化点 | 解决的问题 |
|---|---|---|---|
| 3 | `scripts/eval_mbpp_pass_at_k.py` | pass@k 诊断 + VERDICT | #6 先量化 base 能力，避免白烧 |
| 4 | `scripts/filter_mbpp_by_passrate.py` | 只保留 pass_rate ∈ [0.05, 0.95] 的题 | #3 把"全 0"和"白给"的题过滤掉，每步都有 group 方差 |
| 5 | `scripts/chat_rl.py` 新增 flag | `--reward-mode tiered` + `--group-size 8` + `--log-rollouts-every 50` | #1 阶梯奖励 + #4 更大 group + #5 肉眼看 rollout |

### 阶梯奖励 (`codechat/execution.py:mode="tiered"`)

| 条件 | reward |
|---|---|
| 代码为空 / ast.parse 失败 | 0.00 |
| parse 过 / exec 报错 | 0.05 |
| exec 过 / 0 test 通过 | 0.15 |
| k/n test 通过 (0<k<n) | 0.15 + 0.85·k/n |
| 全部 test 通过 | 1.00 |

阶梯非常粗，只在"完全不会"和"基本能跑"之间打破 0/非0 断层，不会被琐碎的语法进展主导梯度。

### pass@k VERDICT 规则

`eval_mbpp_pass_at_k.py` 结束时会打印判据：

- `pass@1 < 1%` → **"VERDICT: GRPO 会停滞，先强化 base"**，脚本仍会尝试 filter + RL，但 filter 很可能得到空集并 abort。
- `pass@1 ∈ [1%, 5%)` → **"可以用 tiered + group≥8 + 过滤"**，期望值不高。
- `pass@1 ≥ 5%` → **"标准 GRPO 应该能收敛"**，tiered 仍作为稳定器。

### TB 对比

新的 run 名是 `codechat_8b_rl_v2`，不会覆盖旧的 `codechat_8b_rl`。

```bash
./.venv_train/bin/tensorboard --logdir runs/tb
# 打开后对比 rl/reward_mean 曲线：旧 run 应该全程贴 0，新 run 应该能离开 0
# 另外新 run 会把 rl/rollout_best 作为 text 写进 TB，可在 TEXT 面板看模型实际输出
```

---

---

## 参考

- Karpathy, [nanochat](https://github.com/karpathy/nanochat) — 模型与训练循环风格来源
- PyTorch FSDP 官方文档 — <https://pytorch.org/docs/stable/fsdp.html>
- ZeRO 原论文 — 解释了为什么切 optimizer states 是显存命中率最高的一步
- 主仓库 [`README.md`](README.md) — 单卡 2B 路线、数据集、RL 细节
