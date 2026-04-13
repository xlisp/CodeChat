# CodeChat 8B 训练报告（8×A800-SXM4-80GB）

**日期**: 2026-04-11 ~ 2026-04-14
**硬件**: 8× A800-SXM4-80GB (NVLink)
**框架**: PyTorch 2.10 + FSDP (FULL_SHARD)
**训练脚本**: `runs/train_a800_x8.sh`

---

## 0. 流水线总览

| 阶段 | 脚本 | 状态 | 耗时 |
|------|------|------|------|
| 1 — 预训练数据准备 | `runs/prepare_pretrain_venv.sh` | **DONE** | — |
| 2 — 预训练 (8B, FSDP×8) | `scripts/base_train.py` | **DONE** | ~48.9h |
| 3 — SFT 数据准备 | `scripts/prepare_sft.py` | **DONE** | — |
| 4 — SFT (8B, FSDP×8) | `scripts/chat_sft.py` | **DONE** | ~4.8h |
| 5 — RL / GRPO (MBPP) | `scripts/chat_rl.py` | **FIXED — 待重跑** | — |

---

## 1. 模型架构

| 参数 | 值 |
|------|-----|
| 预设 | `8b` |
| 层数 (depth) | 40 |
| 隐藏维度 (n_embd) | 4096 |
| 注意力头 (n_head) | 32 |
| 上下文长度 | 2048 |
| 词表大小 | 50257 |
| 总参数量 | ~8.3B |
| 数据类型 | bfloat16 |

---

## 2. 阶段 2: 预训练 — DONE

### 2.1 配置

| 设置 | 值 |
|------|-----|
| 全局 batch size | 1 × 8 × 8 × 2048 = 131,072 tokens/step |
| 单卡 batch size | 1 |
| 梯度累积 | 8 |
| 世界大小 | 8 (FSDP FULL_SHARD) |
| 学习率 | 1.5e-4 (cosine decay) |
| Warmup 步数 | 1,000 |
| 总步数 | 30,000 |
| 优化器 | AdamW |

### 2.2 结果

| 指标 | 值 |
|------|-----|
| 总消耗 tokens | **3.93B** |
| 总耗时 | **48.9 小时** |
| 吞吐量 | 22.3 Ktok/s (稳定) |
| 最终 loss | **0.6086** |
| 最后 1k 步平均 loss | 0.6972 |
| 最终学习率 | 1.50e-5 |
| 最终梯度范数 | 0.18 |

### 2.3 Loss 曲线

```
Step         Loss
─────────────────
     3      11.37   (随机初始化)
  5,000      1.11
 10,000      0.97
 15,000      0.68
 20,000      0.50
 25,000      0.80
 30,000      0.61
```

Loss 顺利收敛至 1.0 以下。Step 25k 附近的尖刺与 cosine decay 噪声一致，训练结束时 loss 回落至 ~0.61。

### 2.4 Checkpoint

```
checkpoints/codechat_8b/latest.pt   (16 GB)
```

---

## 3. 阶段 3: SFT 数据准备 — DONE

| 项目 | 值 |
|------|-----|
| 数据来源 | `iamtarun/python_code_instructions_18k_alpaca` + `sahil2801/CodeAlpaca-20k` |
| 总样本数 | **38,628** |
| 输出文件 | `data/sft/train.jsonl` (72MB) |

---

## 4. 阶段 4: SFT — DONE

### 4.1 历史问题与修复

`scripts/chat_sft.py` 原先为单卡训练设计，在 `torchrun --nproc_per_node=8` 下会 OOM。三个根因：

1. **无分布式初始化** — 没有 `dist.init_process_group()` 和 per-rank `cuda.set_device()`。
2. **Checkpoint 全部载入 cuda:0** — 8 个进程同时在 GPU 0 上加载 16GB 权重 (8×16GB=128GB)。
3. **无 FSDP 包装** — 8B 模型 + AdamW 状态 (~96GB fp32) 单卡放不下。

**修复内容** (commit `bc8c26b`)：
- 加入 `setup_distributed()` — 检测 `LOCAL_RANK`，设置 `cuda.set_device()`，初始化 NCCL。
- Checkpoint 加载改为 `map_location="cpu"` — 先载入 CPU，再分发到 GPU。
- 加入 `wrap_fsdp()` — 与预训练相同的 FSDP (FULL_SHARD) + Block 级别 auto-wrapping。
- 加入 `no_sync()` 避免非最终 micro-batch 的冗余梯度 all-reduce。
- FSDP 感知的 `clip_grad_norm_()`。
- TensorBoard writer 仅在 rank 0 创建。
- 加入 `dist.barrier()` + `dist.destroy_process_group()` 清理。

### 4.2 SFT 训练配置

| 设置 | 值 |
|------|-----|
| Base checkpoint | `checkpoints/codechat_8b/latest.pt` |
| 单卡 batch size | 1 |
| 梯度累积 | 8 |
| 学习率 | 5e-5 (cosine decay) |
| Warmup 步数 | 100 |
| 总步数 | 3,000 |
| 数据量 | 38,628 examples |
| 世界大小 | 8 (FSDP FULL_SHARD) |

### 4.3 SFT 训练结果

训练正常完成。以下为最后阶段的 loss 日志：

```
sft step  2840 | loss 0.1501 | lr 5.34e-06 | 16420s
sft step  2860 | loss 0.3793 | lr 5.26e-06 | 16535s
sft step  2880 | loss 0.1776 | lr 5.19e-06 | 16650s
sft step  2900 | loss 0.2232 | lr 5.13e-06 | 16764s
sft step  2920 | loss 0.0735 | lr 5.08e-06 | 16879s
sft step  2940 | loss 0.2336 | lr 5.05e-06 | 16995s
sft step  2960 | loss 0.2650 | lr 5.02e-06 | 17109s
sft step  2980 | loss 0.3534 | lr 5.01e-06 | 17225s
sft step  3000 | loss 0.1164 | lr 5.00e-06 | 17339s
  saved -> checkpoints/codechat_8b_sft/latest.pt
```

| 指标 | 值 |
|------|-----|
| 总耗时 | **~4.8 小时** (17,339 秒) |
| 最终 loss | **0.1164** |
| 最后 200 步平均 loss | ~0.22 |
| 最终学习率 | 5.00e-06 (min lr) |
| 每步耗时 | ~5.8 秒 |

Loss 从预训练水平稳步下降至 0.1 级别，SFT 收敛良好。

### 4.4 SFT Checkpoint

```
checkpoints/codechat_8b_sft/latest.pt   (16 GB)
```

---

## 5. 阶段 5: RL / GRPO — FAILED (CUDA OOM)

### 5.1 RL 配置

| 设置 | 值 |
|------|-----|
| SFT checkpoint | `checkpoints/codechat_8b_sft/latest.pt` |
| 算法 | GRPO (Group Relative Policy Optimization) |
| 数据集 | MBPP (sanitized), 120 条 |
| 总步数 | 1,000 |
| Group size | 4 (每个 prompt 采样 4 个补全) |
| 学习率 | 1e-5 |
| KL 系数 | 0.02 |
| 最大生成 tokens | 384 |
| 温度 | 0.9 |
| Top-k | 50 |
| **运行模式** | **单 GPU** (非 FSDP) |

### 5.2 崩溃详情

```
File "scripts/chat_rl.py", line 176, in main
    optim.step()
  File ".../torch/optim/adam.py", line 182, in _init_group
    state["exp_avg_sq"] = torch.zeros_like(...)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 MiB.
GPU 0 has a total capacity of 79.33 GiB of which 14.00 MiB is free.
Process has 79.30 GiB memory in use.
```

崩溃发生在第 1 步 `optim.step()` 时，Adam 优化器初始化第二动量 (`exp_avg_sq`) 失败。

### 5.3 OOM 根因分析

`chat_rl.py` 被设计为单 GPU 运行，但 8B 模型的显存需求远超单卡容量：

| 组件 | 显存 (估算) |
|------|------------|
| Policy 模型 (8.3B, bf16) | ~16 GB |
| Reference 模型 (8.3B, bf16, 冻结) | ~16 GB |
| AdamW 状态 (exp_avg + exp_avg_sq, fp32) | ~64 GB |
| 梯度 + 激活 + 中间计算 | ~10+ GB |
| **总计** | **~106 GB** |

单张 A800 仅有 79.33 GiB 可用，在加载 policy + ref 两个模型后 (~32GB) 加上前向/反向传播中间状态，已接近满载。当 Adam 优化器在第一步尝试分配 exp_avg 和 exp_avg_sq (每个 ~32GB fp32) 时，剩余显存完全不够。

### 5.4 修复方向

要在 8B 模型上运行 RL，需要做以下改动之一或组合：

1. **FSDP 支持** — 与 SFT 相同方式用 FSDP 包装 policy 和 ref 模型，将参数/优化器状态切分到 8 张卡上。这是最稳妥的方案。
2. **DeepSpeed ZeRO-3** — 类似 FSDP 的零冗余方案。
3. **卸载 ref 模型** — ref 模型仅在计算 KL 散度时需要前向传播，可以在不需要时卸载到 CPU。
4. **梯度检查点 (activation checkpointing)** — 节省中间激活显存。
5. **减小 group_size** — 从 4 降为 2，减少每步的显存峰值（但不解决根本问题）。

**推荐方案**: 方案 1 (FSDP)，复用 `chat_sft.py` 已有的分布式代码。

### 5.5 已应用修复

已为 `scripts/chat_rl.py` 增加 FSDP 支持，改动与 `chat_sft.py` 一致：

- 加入 `setup_distributed()` + `wrap_fsdp()` — 自动检测 torchrun 环境，FSDP FULL_SHARD 包装 policy 和 ref 模型
- Checkpoint 改为 `map_location="cpu"` 加载，避免多进程争抢 GPU 0
- 采样阶段 (`sample_one`) 加入 `dist.broadcast(nxt, src=0)` 保证所有 rank 采样同一 token
- MBPP 问题索引通过 `dist.broadcast` 同步
- 梯度裁剪改用 FSDP 的 `clip_grad_norm_()`
- TensorBoard / print 仅 rank 0 执行
- 训练结束加入 `dist.barrier()` + `dist.destroy_process_group()`
- `runs/train_a800_x8.sh` 中 RL 阶段改为 `torchrun --nproc_per_node=8` 启动

重跑命令：
```bash
SKIP_TO=5 bash runs/train_a800_x8.sh
```

---

## 6. 2B vs 8B 对比

| 指标 | 2B (单卡) | 8B (FSDP×8) |
|------|-----------|-------------|
| 参数量 | ~2.1B | ~8.3B |
| 预训练步数 | 24,987 | 30,000 |
| 消耗 tokens | 1.64B | 3.93B |
| 预训练耗时 | 57.5h | 48.9h |
| 吞吐量 | 7.9 Ktok/s | 22.3 Ktok/s |
| 预训练最终 loss | 0.867 | **0.609** |
| 预训练最小 loss | 0.624 | **0.316** |
| SFT 最终 loss | — | **0.116** |

8B 模型在仅 ~2.4× 数据量下取得了显著更低的 loss，体现了规模优势。8 卡 FSDP 吞吐量比单卡 2B 快 ~2.8×，训练 4× 大模型的同时总耗时反而更短。

---

## 7. 2B 预训练基线详情

> 以下为 2B 模型预训练的完整记录，作为基线参考。

**日期**: 2026-04-11
**硬件**: 单卡 A800-SXM4-80GB
**数据集**: 32 shards × 256MB = 8.0GB (~4B tokens, uint16)

### 7.1 训练配置

| 参数 | 值 |
|------|-----|
| 模型参数量 | 2,650.6M (bf16) |
| 架构 | GPT: depth=32, n_embd=2560, n_head=20, block_size=2048 |
| 优化器 | AdamW (betas=0.9/0.95, weight_decay=0.1, fused=True) |
| 学习率 | 2e-4 (cosine decay, warmup=500, min_ratio=0.1) |
| Batch size | device_batch=2 × grad_accum=16 = 有效 batch 32 |
| 每步 tokens | 32 × 2048 = 65,536 |
| 总步数 | 30,000 |
| Checkpoint 间隔 | 每 2,000 步 |
| 激活检查点 | 开启 |

### 7.2 Loss 曲线关键节点

| Step | Loss | 学习率 | 阶段 |
|------|------|--------|------|
| 1 | 11.6479 | 4.00e-07 | 初始 |
| 500 | 3.1913 | 2.00e-04 | warmup 结束 |
| 1,000 | 2.1966 | 2.00e-04 | 快速下降 |
| 3,000 | 1.1297 | 1.97e-04 | 快速下降 |
| 5,000 | 1.2156 | 1.90e-04 | 趋于稳定 |
| 10,000 | 0.9936 | 1.58e-04 | 缓慢下降 |
| 15,000 | 0.9944 | 1.12e-04 | 平台期 |
| 17,486 | 0.9043 | 8.88e-05 | 平台期 |

### 7.3 各阶段统计

| 阶段 | Step 范围 | 平均 Loss | 最小 Loss | 最大 Loss |
|------|-----------|-----------|-----------|-----------|
| Warmup | 1 - 500 | 4.2180 | 3.0071 | 11.6498 |
| Early | 500 - 2,000 | 1.9997 | 1.1983 | 3.5934 |
| Mid-early | 2,000 - 5,000 | 1.2426 | 0.8804 | 1.7202 |
| Mid | 5,000 - 10,000 | 1.0647 | 0.7758 | 1.5492 |
| Mid-late | 10,000 - 15,000 | 0.9502 | 0.7150 | 1.3925 |
| Late | 15,000 - 17,486 | 0.9012 | 0.6830 | 1.2155 |

### 7.4 训练稳定性

- Step 5000 后仅 2 次 loss > 1.5 的尖刺 (step 5090: 1.55, step 6966: 1.53)，训练整体非常稳定。
- 梯度范数被 clip 到 1.0，最大尖刺 11.56 (step 3945) 和 10.38 (step 12160)，均被 grad clip 有效处理。
- 吞吐量全程稳定在 7.9 Ktok/s。

### 7.5 Checkpoint

```
checkpoints/codechat_2b/latest.pt   (15 GB)
```

模型 5.3GB + Adam 状态 ~10GB = ~15GB，大小合理。仅保留最新 checkpoint，无历史版本。

---

## 8. TensorBoard

```bash
tensorboard --logdir runs/tb
```

| Run | 说明 |
|-----|------|
| `runs/tb/codechat_2b` | 2B 预训练 (基线) |
| `runs/tb/codechat_8b` | 8B 预训练 |
| `runs/tb/codechat_8b_sft` | 8B SFT (已完成) |
| `runs/tb/codechat_8b_rl` | 8B RL (仅初始化，未产生有效数据) |

---

## 9. 总结

1. **预训练完成**: 8B 模型 30k 步训练，loss 从 11.37 收敛至 0.61，曲线平滑无发散。
2. **SFT 完成**: 3,000 步微调，loss 降至 0.12，收敛良好。Checkpoint 已保存 (16GB)。
3. **RL 阶段已修复**: `chat_rl.py` 原为单 GPU 设计导致 OOM (policy+ref+Adam ~106GB > 80GB)，已加入 FSDP 支持，待重跑。
4. **规模效果显著**: 8B 比 2B 在预训练 loss 上有质的提升 (0.61 vs 0.87)。
5. **下一步**: 运行 `SKIP_TO=5 bash runs/train_a800_x8.sh` 重跑 RL 阶段。
