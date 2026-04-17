# CodeChat 8B v6 统一模型训练报告

**日期**: 2026-04-16 ~ 2026-04-17

**硬件**: 8× A800-SXM4-80GB (NVLink)

**框架**: PyTorch 2.10 + FSDP (FULL_SHARD), bf16

**相关脚本**: `runs/train_a800_x8_v6.sh`、`scripts/chat_sft.py`、`scripts/chat_rl_funcall.py`

**承接**: v5 funcall RL 成功（`reports/TRAINING_REPORT_8b_funcall_v5.md`），但 v5 成了 funcall 专家 —— ~86% rollouts 输出 `<functioncall>`，丧失普通代码生成能力。v6 目标是 **一个 ckpt 同时处理 code 和 funcall**。

---

## 0. TL;DR

- **v6 是 v5 的能力合并尝试**：joint SFT 在 code + funcall 混合数据上训练，用 `<|system|>` tool block 的有无做推理时的天然 disambiguator。
- **Joint SFT 成功**: 8000 步（≈14.2h），loss 从 2.51 降到 0.2-0.9 区间。funcall 格式匹配率从 SFT 出来就是 **pass@1 = 85.4%、pass@8 = 87.5%**（n=48），与 v5 纯 funcall SFT 的 86.7% 几乎持平 —— 混合数据没有明显拖累 funcall 能力。
- **Funcall RL 饱和**: 跑完全部 283 步（1 epoch），eval 全程 ±0.01 抖动：step 1 = 0.858/0.900，step 270 = 0.864/0.917（pass@1/pass@16）。reward_mean 稳定 0.88-1.0，advantage ≈ 0，梯度接近零。与 v5 走势完全一致。
- **Code 能力丢失**: smoke test 中 "write quicksort" 输出乱码 `quicksort(list [, "list", ...], 3)`，而非有效 Python。**根因是 5.6:1 的数据不平衡**（112k funcall vs 20k code）+ 8000 步 ≈ 7 epochs 的过拟合。
- **Funcall smoke test 有瑕疵**: 问 "Weather in Tokyo?" 输出了 `{"location": "Berlin"}`，格式正确但参数幻觉。
- **结论**: v6 证明了 joint SFT 可以保持 funcall 能力，但 **code 数据量严重不足**。下一步需要平衡数据比例或分阶段训练。

---

## 1. v6 设计动机与 v5 的差距

### v5 的问题

v5 从通用 SFT base 出发，只在 funcall 数据上续训 + RL，结果变成了 funcall 专家：

| 测试 | v5 RL ckpt 行为 |
|---|---|
| "Weather in Tokyo?" (带 tool schema) | `<functioncall> {"name": "get_weather", ...}` — 正确 |
| "Write quicksort" (无 tool schema) | `<functioncall> {"name": "quicksort", ...}` — 错误，把代码请求也当成工具调用 |

v5 RL ckpt 的约 86% rollouts 都输出 `<functioncall>`，无论 prompt 里有没有工具定义。

### v6 的核心设计

| 设计决策 | 原因 |
|---|---|
| **从通用 SFT base 出发**，而非 v5 funcall ckpt | v5 的 funcall-saturated prior 太强，fine-tune 回来成本高 |
| **混合 SFT**: 拼接 code + funcall，shuffle 训练 | 用 `<\|system\|>` tool block 有无做天然消歧 |
| **8000 步**（v5 是 6000） | 两个分布共享 budget，多给 33% 步数 |
| **只做 funcall RL**，不做 code RL | v1/v2 已证明 MBPP 执行 reward 在 8B 上无梯度 |

---

## 2. 流水线与各阶段结果

### 2.0 总览

| 阶段 | 输入 | 输出 | 状态 | 耗时 |
|---|---|---|---|---|
| [1] code SFT 数据准备 | MBPP + Codeforces + the-stack-smol | `data/sft_code/train.jsonl`（20,294 行） | **复用** | — |
| [2] funcall SFT 数据准备 | glaive-function-calling-v2 | `data/sft_funcall/train.jsonl`（112,934 行） | **复用** | — |
| [3] merge + shuffle | cat + shuf | `data/sft_v6/train.jsonl`（133,228 行） | **DONE** | 秒级 |
| [4] joint SFT (FSDP×8) | base = `codechat_8b_sft/latest.pt` | `checkpoints/codechat_8b_sft_v6/latest.pt` | **DONE** | ≈14.2h |
| [5] RL 数据抽取 | glaive 中首轮 functioncall | `data/rl_funcall/{train,eval}.jsonl`（2267/122） | **复用 v5** | — |
| [6] Pre-RL 诊断 | v6 SFT ckpt, lr=0 | 在线 pass@k | **DONE** | ≈2min |
| [7] funcall RL (FSDP×8) | v6 SFT → 283 步 | `checkpoints/codechat_8b_rl_v6/latest.pt` | **DONE** | ≈4.1h |
| [8] dual smoke test | RL ckpt | quicksort + weather | **DONE** | ≈1min |

**总墙钟**: ≈18.5h（SFT 14.2h + diag 2min + RL 4.1h + overhead）

### 2.1 阶段 [3]：数据合并

```
code:    20,294 行 (15.2%)
funcall: 112,934 行 (84.8%)
total:   133,228 行
```

用固定 seed `shuf --random-source=<(yes 42)` 打散，保证两个分布在 epoch 内均匀交替，不会出现前半纯 code 后半纯 funcall 导致的梯度分布跳变。

**⚠ 关键隐患**: code 占比仅 15.2%，在 7 epochs 训练中模型见到 ~142k 个 code 样本 vs ~790k 个 funcall 样本 —— funcall 的曝光量是 code 的 **5.6 倍**。

### 2.2 阶段 [4]：Joint SFT

**超参**:

| 设置 | 值 | 相对 v5 |
|---|---|---|
| Base ckpt | `checkpoints/codechat_8b_sft/latest.pt` | 同（通用 SFT base，非 v5 funcall） |
| 全局 batch | 1 × 8 × 8 × 2048 ≈ 131k tokens/step | 同 |
| 学习率 | 3e-5（cosine, warmup 300 步） | warmup 多 100 步 |
| 总步数 | **8,000**（v5 = 6,000） | +33% |
| 总 token | ≈ 1.05B supervised tokens | v5 ≈ 786M |
| Epochs | ≈ 7 epochs over 133k rows | v5 ≈ 7 epochs over 113k rows |

**SFT loss 曲线** (TensorBoard: `runs/tb/codechat_8b_sft_v6`):

```
step     1 | loss 2.5059 | lr 0.00e+00 |     18s   ← 冷启
step   100 | loss 1.1873 | lr 1.00e-05 |    657s   ← warmup 中
step   300 | loss 1.8734 | lr 3.00e-05 |   1937s   ← warmup 结束，peak lr
step   500 | loss 1.8792 | lr 3.00e-05 |   3085s
step  1000 | loss 0.9528 | lr 2.93e-05 |   6335s
step  2000 | loss 0.3422 | lr 2.74e-05 |  12631s
step  3000 | loss 0.1371 | lr 2.48e-05 |  18930s
step  4000 | loss 0.9068 | lr 2.16e-05 |  25340s
step  5000 | loss 1.0125 | lr 1.81e-05 |  31641s
step  6000 | loss 0.2871 | lr 7.25e-06 |  38315s
step  7000 | loss 0.4300 | lr 4.11e-06 |  44700s
step  8000 | loss 0.2054 | lr 3.00e-06 |  51087s   ← 终点
```

**grad_norm** 全程稳定在 0.67-0.85，无爆炸或坍塌：

```
step  1000: 0.852
step  2000: 0.699
step  3000: 0.676
step  4000: 0.738
step  5000: 0.766
step  6000: 0.777
step  7000: 0.727
step  8000: 0.852
```

**Loss 分析**:
- 前 300 步 warmup 期间 loss 从 2.5 快速降到 ~1.2
- Step 300-1000: lr 到达峰值后 loss 继续下降到 ~1.0
- Step 1000-3000: loss 稳步降至 0.1-0.9 区间
- Step 3000-8000: loss 在 0.1-1.0 区间波动，中位值约 0.5-0.7，无进一步下降趋势
- 最终 loss 能到 0.2 说明模型拟合度不错，但单步 loss 抖动大（0.06-2.02），与 v5 行为一致 —— funcall 样本长度和难度差异大

**观察**: Loss 在 step 3000 后基本收敛。后 5000 步贡献的是更精细的拟合和 cosine lr 衰减的正则效应，但也是 code 分布被 funcall 不断覆写的窗口。

### 2.3 阶段 [6]：Pre-RL 诊断

```
[eval] step 1 | pass@1 0.8542 | pass@8 0.8750 | n=48
```

对比 v5 诊断（纯 funcall SFT 后）：

| 指标 | v5 SFT | v6 joint SFT | Δ |
|---|---|---|---|
| pass@1 | 0.867 | 0.854 | -0.013 |
| pass@8 | 0.875 | 0.875 | 0.000 |

**结论**: Joint SFT 对 funcall 的格式匹配率只掉了 1.3 个百分点，几乎可以忽略。混合数据没有干扰 funcall 学习 —— `<|system|>` tool block 是一个足够强的条件信号。

### 2.4 阶段 [7]：Funcall RL

**超参（与 v5 完全一致）**:

| 设置 | 值 |
|---|---|
| 算法 | REINFORCE with baseline (`advantage = r - r.mean()`) |
| num-samples | 16 per rank → 8 × 16 = **128 rollouts/step** |
| num-epochs | 1 → num_steps = 2267/8 = **283** |
| lr | 1e-5 × init_lr_frac=0.05 → 起点 5e-7, cosine → 0 |
| max-new-tokens | 256 |
| temperature / top-k | 1.0 / 50 |
| eval-every / eval-examples | 30 / 200 (实际 n=120 因整除对齐) |

**Eval 曲线** (TensorBoard: `runs/tb/codechat_8b_rl_v6`):

```
step     1 | pass@1 0.8583 | pass@16 0.9000 | n=120
step    30 | pass@1 0.8536 | pass@16 0.9083
step    60 | pass@1 0.8516 | pass@16 0.9000
step    90 | pass@1 0.8630 | pass@16 0.9083   ← pass@1 微涨
step   120 | pass@1 0.8573 | pass@16 0.9083
step   150 | pass@1 0.8635 | pass@16 0.9083
step   180 | pass@1 0.8615 | pass@16 0.9083
step   210 | pass@1 0.8594 | pass@16 0.9000
step   240 | pass@1 0.8583 | pass@16 0.9083
step   270 | pass@1 0.8635 | pass@16 0.9167   ← pass@16 最高点
```

| 指标 | Step 1 | Step 270 | Δ |
|---|---|---|---|
| pass@1 | 0.858 | 0.864 | **+0.006** |
| pass@16 | 0.900 | 0.917 | **+0.017** |

**RL 全程抖动 ±0.01，无明显趋势**。与 v5（step 60 即 peak，之后 flat）一致。

**Training reward**:

```
step    1 | reward_mean=0.926 | reward_std=0.052
step   30 | reward_mean=0.801 | reward_std=0.036
step   60 | reward_mean=0.995 | reward_std=0.022
step   90 | reward_mean=0.786 | reward_std=0.005   ← 低 reward 但 std 极小 = 难题
step  120 | reward_mean=1.000 | reward_std=0.000
step  150 | reward_mean=0.977 | reward_std=0.050
step  180 | reward_mean=0.882 | reward_std=0.065
step  210 | reward_mean=1.000 | reward_std=0.000
step  240 | reward_mean=0.863 | reward_std=0.082
step  270 | reward_mean=1.000 | reward_std=0.000
step  283 | reward_mean=0.977 | reward_std=0.074
```

Reward 波动来自不同难度的 batch，而非学习进展。大量 step reward=1.0/std=0.0 说明 batch 内所有 128 个 rollouts 都拿到满分 —— **模型已经把这些样本做穿了**。

**Advantage 分析**:

```
step    1 | adv_abs_mean=0.046
step   30 | adv_abs_mean=0.023
step   60 | adv_abs_mean=0.010
step   90 | adv_abs_mean=0.002   ← 几乎无梯度
step  120 | adv_abs_mean=0.000
step  150 | adv_abs_mean=0.038   ← 偶发难题
step  180 | adv_abs_mean=0.038
step  270 | adv_abs_mean=0.000
```

Advantage 大部分时间为 0，REINFORCE 梯度 `∇ = advantage × ∇log π` ≈ 0。RL 本质上是空转。

**Tier 分布** (283 步中各 tier 出现的次数):

| Tier | 出现 step 数 | 总权重 | 含义 |
|---|---|---|---|
| full_match | 269 / 283 | 258.9 | batch 的主体 rollout 拿到满分 |
| no_tag | 35 | 5.3 | 未输出 `<functioncall>` 标签 |
| bad_json | 22 | 16.1 | JSON 格式错误（引号混用等） |
| wrong_name | 9 | 0.6 | 函数名不匹配 |
| partial_0.00 | 3 | 1.1 | 名对参数全错 |
| partial_0.50 | 2 | 0.1 | 名对参数半对 |
| name_only | 1 | 0.9 | 仅名字匹配 |

**95% 的步都是 full_match 主导**。失败模式集中在 bad_json（与 v5 一致的 JSON-in-JSON 引号问题）和偶发的 no_tag。

**Rollout 样例** (来自日志):

```
[step  50] gt: check_email_spam(email_subject="You've won...", email_body="Click on...")
           all 16 samples reward=0.15 → bad_json (单引号 vs 双引号)

[step 100] gt: get_random_quote()
           14/16 reward=1.0, 1/16 reward=0.0, 1/16 reward=0.35

[step 200] gt: get_random_quote()
           16/16 reward=1.0 — perfect

[step 250] gt: get_random_joke()
           16/16 reward=1.0 — perfect
```

### 2.5 阶段 [8]：Dual Smoke Test

**Code task** — "Write a Python implementation of quicksort."

```
quicksort(list [, "list", "tuple", "listarray", "int", "float"], 3)
```

❌ **失败**。输出既不是 Python 代码也不是 funcall JSON，是一段乱码。模型丧失了代码生成能力。

**Funcall task** — "Weather in Tokyo?"

```
[assistant] <functioncall> {"name": "get_weather", "arguments": '{"location": "Berlin", "unit": "celsius"}'}
[function_response] {"location": "Berlin", "temperature": 22, ...}
[assistant] The current weather in Tokyo is $22.
```

⚠ **格式正确但参数幻觉**。问的是 Tokyo，生成的是 Berlin。第二轮答复照搬了 mock response 但称之为 Tokyo —— 前后矛盾。

---

## 3. TensorBoard 分析

### 3.1 SFT Loss (`runs/tb/codechat_8b_sft_v6`)

共 8000 个 data points。

**关键观察**:

1. **收敛速度**: Loss 在前 1000 步从 2.5 降到 ~1.0，前 3000 步降到 ~0.5 中位值。之后 5000 步 loss 在 0.1-1.0 区间抖动无明显下降。
2. **Grad norm 稳定**: 全程 0.67-0.85，无 spike。说明混合数据的梯度分布合理，shuffle 起到了作用。
3. **过拟合信号**: Step 3000 后 loss 不再下降但也没上升，单步可低至 0.065（step 6440）、0.095（step 6300）—— 这些极低值可能是模型已经 memorize 了重复出现的 funcall 样本。
4. **Lr schedule**: Cosine 从 3e-5 衰减到 3e-6，最后 2000 步 lr 已经很小，更新幅度有限。

### 3.2 RL Eval (`runs/tb/codechat_8b_rl_v6`)

10 个 eval checkpoint，每 30 步一次。

**pass@1**: 全程 0.852-0.864，标准差 0.004。**统计学上无显著变化**。
**pass@16**: 全程 0.900-0.917，标准差 0.006。末尾 0.917 是最高值但仍在抖动范围内。

### 3.3 RL Reward / Advantage

- `reward_mean`: 大部分 step 在 0.88-1.0，中位值 ~0.96
- `reward_std`: 大量 step 为 0.0（batch 内完全一致），说明 policy 对大部分 prompt 已经 collapsed 到确定性输出
- `advantage_abs_mean`: 趋势下降，中后期频繁为 0.0 → REINFORCE 更新量为零
- `loss_pg`: 全程 ≈ -0.0000，偶发 -0.14（step 160）、-0.04（step 260）

**图示**（文本 ASCII）:

```
pass@1  0.87 |
        0.86 |    .  .     . .  .
        0.85 | .    .   .       .   .
        0.84 |
             +---+---+---+---+---+---+---+---+---+
               1  30  60  90 120 150 180 210 240 270

reward  1.00 |  . .   .    .   .  .    . .   .
        0.90 |.          .   .       .     .    .
        0.80 |    .     .
        0.70 |
             +---+---+---+---+---+---+---+---+---+
               1  30  60  90 120 150 180 210 240 270 283
```

### 3.4 与 v5 对比

| 指标 | v5 (funcall-only SFT → RL) | v6 (joint SFT → RL) |
|---|---|---|
| SFT pre-RL pass@1 | 0.867 | 0.854 |
| SFT pre-RL pass@8 | 0.875 | 0.875 |
| RL start pass@1 | 0.852 | 0.858 |
| RL start pass@16 | 0.908 | 0.900 |
| RL peak pass@1 | 0.860 (step 60) | 0.864 (step 90/150/270) |
| RL peak pass@16 | 0.908 (step 1) | 0.917 (step 270) |
| RL 有效步数 | ~60（之后 flat） | ~全程 flat |
| Code 能力 | ❌ (全输出 funcall) | ❌ (输出乱码) |
| 主要失败 tier | bad_json (10.3%) | bad_json + no_tag |

**两版 funcall 性能基本持平**，差异在噪声范围内。v6 的 joint SFT 没有帮助也没有伤害 funcall RL。

---

## 4. 问题诊断：Code 能力为何丢失

### 4.1 数据不平衡

```
funcall: 112,934 行 (84.8%)  ← 约 5.6× code
code:     20,294 行 (15.2%)
```

8000 步 × 131k tokens/step ÷ (133k 行 × avg ~7.5k tokens/row) ≈ **7 epochs**。每个 code 样本被看到约 7 次，但每个 funcall 样本也被看到约 7 次 —— 总曝光量 funcall 是 code 的 5.6 倍。

在 cross-entropy loss 中，funcall 梯度的 **累积量** 远大于 code。模型参数向 funcall 分布倾斜。

### 4.2 无独立验证

v6 在 SFT 期间没有监控 code 能力（只看 total loss）。如果有 code-only 验证集，可能在 step 3000-4000 就能看到 code perplexity 开始上升。

### 4.3 Smoke test 不在 SFT 后运行

Stage 8 只在 RL 之后运行。如果在 Stage 4（SFT）完成后立刻做 quicksort smoke test，就能更早发现 code 退化，避免浪费 4h RL 时间。

---

## 5. 各 Checkpoint 推荐

| Checkpoint | 路径 | 用途 |
|---|---|---|
| Joint SFT | `checkpoints/codechat_8b_sft_v6/latest.pt` | 如果 code 没彻底丢，可从这里做 code-only 补训 |
| RL best | `checkpoints/codechat_8b_rl_v6/step_000270.pt` | Funcall pass@16 = 0.917 最高点 |
| RL final | `checkpoints/codechat_8b_rl_v6/latest.pt` (= step_000283) | 同上基本等价 |

**注**: RL 对 funcall 的增益极小（+0.6% pass@1），SFT ckpt 本身已经够用。如果后续要在 code 方向修复，从 SFT ckpt 出发更合理，避免 RL 引入的 funcall-specific 偏移。

---

## 6. 下一步建议

### 6.1 修复 code 能力

**方案 A — 数据均衡重训**:
- 将 code 数据上采样 4-5× 到 ~100k 行，使 code:funcall ≈ 1:1
- 来源：增加 the-stack-smol 的采样量，加入 HumanEval/APPS 等更多代码数据集
- 从通用 SFT base 重新跑 joint SFT（不从 v6 ckpt 续训，避免继承 funcall 偏移）
- 预计 SFT 步数可降到 6000（1:1 比例下收敛更快）

**方案 B — 分阶段训练**:
- Stage 1: 用大规模 code 数据做 SFT（保持 code 能力）
- Stage 2: 在 code SFT ckpt 上用 funcall 数据做短程续训（2000-3000 步），添加 code replay（混入 10-20% code 数据防止遗忘）
- Stage 3: funcall RL（同 v5/v6）

**方案 C — 在 v6 SFT ckpt 上做 code 补训** (最省时):
- 从 `checkpoints/codechat_8b_sft_v6/latest.pt` 出发
- 只用 code 数据训 2000 步，lr 1e-5（小心不要反过来破坏 funcall）
- 每 500 步验证 funcall pass@1 + code quicksort
- 风险：双向拉锯，可能找不到均衡点

### 6.2 SFT 期间添加分域监控

下一版 `scripts/chat_sft.py` 应支持：
- 每 N 步评估 code-only 验证 loss 和 funcall-only 验证 loss（分开算）
- 或跑一次 quicksort + funcall smoke test
- 用 TensorBoard 分别追踪 `sft/loss_code` 和 `sft/loss_funcall`

### 6.3 RL 考虑跳过或大幅缩减

v5 和 v6 两次 RL 都证明：**从 85%+ 的 SFT 起点做 REINFORCE 几乎无收益**。如果下一版 SFT pass@1 仍 >85%:
- 考虑直接跳过 RL（省 4h GPU 时间）
- 或只做 30 步 diagnostic 确认无退化
- 如果想从 RL 获得更多收益，需要更有挑战性的 eval set（当前 122 题太简单、模型已饱和）

---

## 7. 时间线

```
2026-04-16 18:44   SKIP_TO=3, 启动 v6 pipeline
2026-04-16 18:45   [3/8] merge code+funcall → 133,228 行
2026-04-16 18:45   [4/8] joint SFT 启动 (FSDP x8)
2026-04-17 08:59   [4/8] SFT 完成 (8000 步, 51087s ≈ 14.2h)
2026-04-17 08:59   [5/8] RL data 复用 v5
2026-04-17 08:59   [6/8] pre-RL diagnostic: pass@1=0.854, pass@8=0.875
2026-04-17 09:04   [7/8] funcall RL 启动
2026-04-17 13:09   [7/8] RL 完成 (283 步, 14744s ≈ 4.1h)
2026-04-17 13:10   [8/8] smoke test: funcall ✓(有瑕疵), code ✗
2026-04-17 13:10   Pipeline 完成
```

**总计**: 启动到完成约 **18.5 小时**（SFT 14.2h 占 77%）。
