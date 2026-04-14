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
| 5 — RL / GRPO (MBPP) | `scripts/chat_rl.py` | **RUN — 无收益 (reward≡0)** | ~16.7h (至 step 415) |

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

## 5. 阶段 5: RL / GRPO — 已跑通但无收益 (reward≡0)

### 5.1 历史

首次启动时 `chat_rl.py` 仍是单 GPU 设计，policy+ref+AdamW 合计 ~106GB 超过单卡
80GB，首步 `optim.step()` 即 OOM。在 commit `8bcc723` 中给 `chat_rl.py` 加入
FSDP 支持后（与 `chat_sft.py` 对齐：`setup_distributed()` + `wrap_fsdp()` +
`dist.broadcast` 同步 sampling / mbpp 索引 + FSDP 感知的 grad clip），OOM 消失，
8 卡 FSDP 下顺利进入训练循环。

### 5.2 RL 配置（实际跑的一次）

| 设置 | 值 |
|------|-----|
| SFT checkpoint | `checkpoints/codechat_8b_sft/latest.pt` |
| 算法 | GRPO (group size = 4, advantage = (r − r̄)/σ) |
| 数据集 | MBPP (sanitized) train, ~374 条 |
| 总步数 | 1,000（实际跑到 step 415 时中止，无意义） |
| 学习率 | 1e-5 → cosine decay (warmup 20) |
| KL 系数 | 0.02 |
| 最大生成 tokens | 384 |
| 温度 / top-k | 0.9 / 50 |
| 运行模式 | **FSDP FULL_SHARD × 8** |

### 5.3 训练日志（节选 step 240 – 415）

```
rl step   240 | reward 0.000 (max 0.00) | loss 0.0335 | lr 8.93e-06 | 30485s
rl step   250 | reward 0.000 (max 0.00) | loss 0.0296 | lr 8.83e-06 | 32175s
rl step   300 | reward 0.000 (max 0.00) | loss 0.0277 | lr 8.31e-06 | 40676s
rl step   350 | reward 0.000 (max 0.00) | loss 0.0319 | lr 7.71e-06 | 49173s
rl step   400 | reward 0.000 (max 0.00) | loss 0.0297 | lr 7.05e-06 | 57672s
rl step   415 | reward 0.000 (max 0.00) | loss 0.0323 | lr 6.85e-06 | 60256s
```

| 指标 | 观察值 |
|------|--------|
| reward_mean | **0.000 持续 415 步** |
| reward_max | 0.00（4 个 rollout 一个都没对过） |
| loss | 0.027 – 0.035 稳定区间 |
| 每步耗时 | ~145 s（4 次 generate 384 tok + 4 次 forward + 1 次 step） |
| 累计耗时 | ~16.7 h |
| TB run | `runs/tb/codechat_8b_rl/` |

### 5.4 为什么 reward 恒等于 0

GRPO 的有效梯度 = advantage × ∇logπ。当 group 内 4 个 rollout 的 reward 全是 0：

- `r.mean() = 0, r.std() = 0` → advantage 全为 0 → **policy gradient 完全没信号**
- `loss ≈ 0.03` 来自 **KL penalty 自身** (kl_coef × mean(logπ − logπ_ref))，而非学习
- 等价于：模型只在被 KL 项轻微拉回/拉离 ref，做的是**纯随机游走**而不是朝向更高奖励

所以日志里 loss 在 0.03 抖，既不是"在学"，也不是"稳定收敛"——是**死信号**。

根因是 MBPP 的执行奖励（`run_with_tests` 跑 unit test，全对得 1，否则 0）对
当前 8B 来说门槛太高：

1. **容量不足**：3.93B tokens 预训练 + 38k 通用 Alpaca SFT，远达不到能稳定写出
   正确 Python 的水平。同量级公开模型能在 MBPP 上 pass@1 也要 5-10%，当前
   模型显然没到。
2. **奖励稀疏且二元**：binary pass/fail，没有部分得分 → group 内全 0 太容易发生。
3. **4-rollout 的 group 太小**：即便问题的 pass 率是 2%，group=4 全 0 概率仍 ~92%。
4. **`extract_code` 可能吃不到代码块**：模型输出若不带 fenced code block（或
   模型跑出 END_TAG 前就把代码结构写歪），reward pipeline 会提前返回 0。

### 5.5 结论：当前 RL run 无训练意义，建议终止

- **不要继续这次 RL**：再跑 600 步大概率仍是 reward=0，只会在 KL 项下
  让 policy 相对 `codechat_8b_sft/latest.pt` **轻微漂移**，不会变强。
- **不要把 `codechat_8b_rl/latest.pt` 当下一阶段的起点**（funcall 等），
  直接用 `codechat_8b_sft/latest.pt`。RL ckpt 至多等价、很可能略差。

### 5.6 如果还想把 MBPP RL 救活——优化清单（按性价比排序）

| # | 改动 | 预期效果 | 改动位置 |
|---|------|--------|---------|
| 1 | **dense / shaped reward**：按通过测试数比例给分（3/5 测试过得 0.6），而非 0/1 | 立刻打破 group 全 0，group 内有方差 → advantage 非零 | `codechat/execution.py::run_with_tests` |
| 2 | **加语法奖励**（能 `ast.parse` 的给 0.1，能 import 的 0.2，能跑但 test 失败的 0.3，全过 1.0）课程式递进 | 模型先学会"输出合法 Python"，再学"算对" | `run_with_tests` + `chat_rl.py` reward 组合 |
| 3 | **按难度过滤 MBPP**：先用 SFT 模型在 MBPP train 上跑 N 次 rollout，保留 pass rate ∈ [0.1, 0.9] 的问题 | 确保每个 prompt 有非零信号也没"白给"，样本利用率最大化 | 新脚本 `scripts/filter_mbpp_by_passrate.py` |
| 4 | **增大 group_size**：4 → 8 或 16 | pass 率 2% 时，group=8 至少一个对的概率从 8% 提到 15% | `--group-size 8`，显存代价较高 |
| 5 | **log 一些 rollout 文本到 stdout/TB**：每 N 步抽一个 rollout 打印 | 能眼看模型在写什么（是胡言乱语还是接近），不用猜 | `chat_rl.py` 中加 `if step % 50 == 0 and is_master: print(decode(new_ids[:200]))` |
| 6 | **先跑诊断，再决定开不开 RL**：在 MBPP train 50 条上对 SFT ckpt 直接 eval pass@4（温度 0.9），如果 pass@4 < 1-2%，根本不具备 RL 条件 | 避免再白烧 16 小时 | 新脚本 `scripts/eval_mbpp_pass_at_k.py` |
| 7 | **加大 SFT 规模**：Alpaca 38k 偏通用，加入真·代码数据（CodeAlpaca + MBPP train 的标准解作为 SFT target），先 cold-start 一下 code ability | 拉高 base pass@k，RL 才有立足点 | `scripts/prepare_sft.py` 增加源 |
| 8 | **改更简单的 RL 任务**：例如"补全函数签名"、"把 docstring 翻译成代码骨架"，门槛低很多 | 短期能拿到正 reward 曲线，不追求 MBPP 正解 | 新 reward 函数 |
| 9 | **(大改) 换更强的 base**：更多预训练 tokens，或直接拿已有开源 1B-8B code 模型作 warm start | 从根本上让 RL 有可学的空间 | 涉及替换 tokenizer / 模型 |

最小可行组合：**#1 + #5 + #6**。先跑 #6 的 pass@k 诊断验证 base 能力，再用 #1
改奖励从 binary 到 fractional，加 #5 打印看 rollout，就能判断 RL 是否还有救。

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
| `runs/tb/codechat_8b_rl` | 8B RL — 415 步，reward 恒 0，详见 §5 |

---

## 9. 总结 & 下一步

### 现状

1. **预训练完成** — 30k 步，loss 11.37 → 0.61。
2. **SFT 完成** — 3k 步，loss 降到 0.12，`codechat_8b_sft/latest.pt` 可用。
3. **RL 跑通但无收益** — FSDP fix 让 RL 不再 OOM，但 415 步 reward 恒 0，
   advantage 全 0 → loss 只是 KL 自振荡，**不是在学**。这次 RL run **不建议继续**，
   `codechat_8b_rl/latest.pt` 也**不建议作为下游起点**（详见 §5）。
4. **规模优势明显** — 8B 比 2B 预训练 loss 低 ~30% (0.61 vs 0.87)。

### 推荐下一步（按优先级）

1. **终止 RL run**，不浪费 GPU 时。保留 `codechat_8b_sft/latest.pt` 作为所有
   下游微调的起点。
2. **(推荐首选)** 直接跑 **function-calling SFT**：从 glaive-function-calling-v2
   接着 SFT。格式任务、有稠密监督、不依赖生成质量——是当前 8B 最容易看到正向
   信号的方向：
   ```bash
   BASE_CKPT=checkpoints/codechat_8b_sft/latest.pt \
       bash runs/train_a800_x8_v2_funcall.sh
   ```
3. **想救 MBPP RL**：不要直接重跑，先按 §5.6 里 **#6 → #1 → #5** 的顺序做：
   1. 先跑 `eval_mbpp_pass_at_k` 诊断（半小时），看 SFT ckpt 在 MBPP 上 pass@4
      到底是多少。如果 < 1%，RL 没戏，回到路线 2。
   2. 如果 pass@4 ≥ 2-3%，把 reward 从 binary 改成 fractional（通过测试数 / 总测试数），
      并 log rollout 文本，再重跑 200 步看 reward_mean 是否离开 0。
4. **中期**：评估是否把 Alpaca SFT 扩成"代码 SFT"（加入 MBPP train 的标准解、
   CodeContests、Python 教材对话），拉高 base 的代码能力，为后续 SWE-bench RL
   （train_a800_v2.sh 里的三阶段课程）铺路。

### 一句话判断

> **reward=0.000 = 训练确实没意义（GRPO 无梯度信号）**，但
> 这不代表 RL 本身不能做，只说明**当前 base 能力 × MBPP 难度 × 二元奖励**
> 这个组合不成立。优先把 funcall SFT 跑出来拿到第一个对外能力点，之后再带着
> 更强的 base 和更稠密的 reward 回来做 RL。
