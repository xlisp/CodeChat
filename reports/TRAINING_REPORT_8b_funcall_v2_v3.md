# CodeChat 8B Function-Calling + MBPP RL 训练报告（v2 复盘 / v3 设计 / v3 复盘）

**日期**: 2026-04-14 ~ 2026-04-15
**硬件**: 8× A800-SXM4-80GB (NVLink)
**框架**: PyTorch 2.10 + FSDP (FULL_SHARD)，bf16
**相关脚本**: `runs/train_a800_x8_v2_funcall.sh`（v2）、`runs/train_a800_x8_v3_funcall.sh`（v3）、`runs/train_a800_x8_v5_funcall.sh`（v5，v3 失败后的重构）

---

## 0. TL;DR

- **v2 funcall SFT 阶段**: 成功完成 4000 步，loss 从 2.13 收敛至 0.5 ~ 1.0 区间，ckpt 保存到 `checkpoints/codechat_8b_funcall/latest.pt`。
- **v2 RL 诊断阶段**: 在 base ckpt `checkpoints/codechat_8b_sft` 上评测 50 题 × 8 采样 = 400 样本，**pass@1 = 0.00%、pass@8 = 0.00%、parseable = 1.75%、runnable = 1.25%**。过滤后无一题满足 `pass_rate ∈ [0.05, 0.95]`，RL 无法启动。
- **根因**: 不是 reward 设计或 group_size 的问题，而是基座模型的代码生成能力过低——连合法 Python 都写不出来。
- **v3 方案**: 在 funcall SFT 之前插入 “code SFT 续训” 阶段，数据扩展为 MBPP（非 train 切分）+ Codeforces-Python + the-stack-smol（docstring 子集）。诊断改为评测 **sft_code ckpt**，GATE 阈值 parseable ≥ 50% 且 pass@8 ≥ 10%，达标才进 RL。
- **v3 实际结果（2026-04-15 运行）**: ❌ **没达到预期，RL 仍然没跑起来**。code SFT (2000 步) 和 funcall SFT (4000 步) 都顺利完成，但 funcall SFT 的 loss 曲线与 v2 **逐 step 几乎完全重合**（step 20: 2.15 vs 2.13、step 4000: 0.633 vs 0.629），说明 code-SFT 对下游没有可测量的增益。Stage 6 过滤 MBPP 仍然报 `no problems fell in pass_rate in [0.05, 0.95]; falling back to mean_tiered >= 0.1`——且 fallback 也为空，GRPO 从未启动。v3 等价于 "v2 + 一次不起作用的代码续训"。
- **结论**: "先加一段代码 SFT 再 SFT/RL" 这条路在 8B 基座上不通，**问题不是代码数据不够，而是 MBPP 的 instruction→完整函数分布与我们的 SFT 数据分布差得太远，同时 MBPP 的二值 pass 奖励在弱基座上天然无信号**。后续工作切换到 v5 方案：SFT 和 RL 使用同一分布（glaive-function-calling-v2），奖励改为 format-first 阶梯式，绕开 MBPP RL。详见 `runs/train_a800_x8_v5_funcall.sh`。

---

## 1. v2 训练流水线与结果

### 1.1 流水线（`runs/train_a800_x8_v2_funcall.sh`）

| 阶段 | 说明 | 状态 |
|------|------|------|
| 1 — funcall SFT 数据准备 | `glaiveai/glaive-function-calling-v2`，~113k 多轮对话 | **DONE** |
| 2 — funcall SFT（FSDP×8） | 从 `codechat_8b_sft` 续训，4000 步 | **DONE** |
| 3 — MBPP pass@k 诊断 | 评测 base SFT ckpt | **FAIL (reward=0)** |
| 4 — 按 pass rate 过滤 MBPP | 保留 group 内有方差的题 | **FAIL (0 题)** |
| 5 — 优化版 GRPO | tiered reward + group_size=8 | **未启动** |

### 1.2 阶段 2：funcall SFT

**数据**: `data/sft_funcall/train.jsonl`，112,934 条样本（跳过 26 条），格式为多轮 `<|system|>/<|user|>/<|assistant|>/<|function_response|>`，loss 只计 assistant 段 token。

**超参**:

| 设置 | 值 |
|------|-----|
| Base ckpt | `checkpoints/codechat_8b_sft/latest.pt` |
| 全局 batch | 1 × 8 × 8 × 2048 ≈ 131k tokens/step |
| 学习率 | 3e-5（cosine, warmup 200） |
| 总步数 | 4,000 |
| 耗时 | ~25,482s（≈ 7 小时 4 分） |

**Loss 曲线（关键采样）**:

```
step    20 | loss 2.1333 | lr 3.00e-06 | 152s
step   100 | loss 0.6546 | lr 1.50e-05 | 663s
step   200 | loss 0.5899 | lr 3.00e-05 | 1296s
step  1000 | loss ~0.6   | (lr peak 后开始衰减)
step  3500 | loss 0.5782 | lr 4.14e-06 | 22300s
step  4000 | loss 0.6291 | lr 3.00e-06 | 25482s
```

**结论**: SFT 收敛正常，loss 稳定在 0.5 ~ 1.0 区间震荡，典型的多轮 SFT 尾段表现。ckpt 已保存到 `checkpoints/codechat_8b_funcall/latest.pt`。TB 日志 `runs/tb/codechat_8b_funcall/`。

### 1.3 阶段 3：MBPP pass@k 诊断

**重要**：脚本有意用 `RL_BASE_CKPT=checkpoints/codechat_8b_sft/latest.pt`（通用 SFT ckpt，**非** funcall ckpt），因为 funcall SFT 专门学 `<functioncall>` JSON 格式，会稀释 RL 所需的代码能力。

**结果**（50 题 × 8 采样 = 400 样本）:

| 指标 | 数值 |
|------|-----:|
| pass@1（平均每样本） | **0.00%** |
| pass@8（每题有任一样本通过） | **0.00%** |
| 平均 tiered reward | 0.0021 |
| runnable（能执行不崩） | 1.25% |
| parseable（AST 能解析） | 1.75% |

脚本自带的 VERDICT：

> pass@1 < 1%. GRPO with binary/fractional reward will likely stall. Use `--reward-mode tiered` AND filter to problems with some signal, OR strengthen the base model (more SFT) first.

### 1.4 阶段 4：过滤失败

- 过滤条件 `pass_rate ∈ [0.05, 0.95]`：**0 题命中**
- Fallback `mean_tiered >= 0.1`：**仍然 0 题**
- 日志："still nothing after fallback, base model can't even produce parseable code. Improve the SFT stage before running RL."

RL 阶段未启动。

---

## 2. 根因分析

### 2.1 为什么 base ckpt 的代码能力这么低？

- `codechat_8b_sft` 是 2026-04-13 产出的通用 SFT ckpt，数据仅来自 `iamtarun/python_code_instructions_18k_alpaca` + `sahil2801/CodeAlpaca-20k`，合计 38,628 条——对于一个 8B 从头预训练的模型而言，覆盖面偏窄。
- 预训练数据的代码占比未审计，预训练 30k 步（~3.9B tokens）总量也偏小。`the-stack` 级别的广覆盖代码语料没有明确引入。
- 结果：模型见过 “alpaca 风格短函数”，但不会在 MBPP 这种需要 prompt→ 完整独立函数的设定下稳定地写出可解析的代码（parseable 仅 1.75%）。

### 2.2 为什么 v2 的 RL 优化方案没救回来？

v2 相对旧 RL 引入了三个补丁：

1. tiered reward（阶梯式）：`syntax 错 0 / 运行报错 0.05 / 能跑 0 test 过 0.15 / k/n 过 0.15+0.85*(k/n) / 全过 1.00`
2. group_size 从 4 升到 8
3. 过滤出 group 内有方差的题

这三条都是 **在 base 已经会写代码的前提下** 给弱信号加放大器。但当 parseable 只有 1.75%、runnable 只有 1.25% 时，group 内绝大多数都是 `syntax 错 → reward=0`，方差依然趋近 0，过滤后也没题留下。换句话说：**RL 需要 base 有最低限度的可执行代码能力才能起飞，tiered/filter/group_size 只能“拉大信号”，不能“凭空制造信号”。**

---

## 3. v3 方案

### 3.1 设计原则

> 先把 base 的 MBPP parseable 拉到 ≥ 50%、pass@8 ≥ 10%，再谈 RL。

### 3.2 新数据集

新增脚本 `scripts/prepare_sft_code.py`，输出 `data/sft_code/train.jsonl`，聚合三个源：

| 源 | 来源 | 理由 | 上限 |
|---|---|---|---|
| MBPP sanitized（非 train） | `google-research-datasets/mbpp` | 高质量 (prompt, code, tests) 三元组 | ~600 条（test+validation+prompt） |
| Codeforces-Python | `MatrixStudio/Codeforces-Python-Submissions` | 题面→解题代码，风格多样 | 20,000 条（可调 `CODE_MAX_CF`） |
| the-stack-smol | `bigcode/the-stack-smol` data_dir=python | 真实 Python 工程代码，docstring → 模块 | 20,000 条（可调 `CODE_MAX_STACK`） |

**关键防泄漏约定**: `scripts/eval_mbpp_pass_at_k.py` 和 `scripts/chat_rl.py` 默认都在 MBPP sanitized **train** 切分上评测 / 训练；因此 code-SFT **只用 test/validation/prompt 三个切分**，把 train 切分留给诊断 GATE 和 RL，保证诊断数字是真实泛化能力而不是训练集记忆。

数据质量约束：
- the-stack-smol 只保留能 `ast.parse` 且有 module docstring（≥ 20 字符）的文件
- code 长度裁剪到 ≤ 4000 字符（避免 block_size=2048 下严重截断）
- 单条 supervised token 数 < 8 的直接丢弃

### 3.3 新流水线（`runs/train_a800_x8_v3_funcall.sh`）

| 阶段 | 说明 | 输出 |
|------|------|------|
| 1 — code SFT 数据 | `scripts.prepare_sft_code` | `data/sft_code/train.jsonl` |
| 2 — **code SFT 续训** | base=`codechat_8b_sft`，lr 2e-5，2000 步 | `checkpoints/codechat_8b_sft_code/latest.pt` |
| 3 — pass@k 诊断（新） | **评测 sft_code ckpt**，GATE = parseable ≥ 50% & pass@8 ≥ 10% | `data/mbpp_passrate.jsonl` |
| 4 — funcall SFT 数据 | 复用 v2 | `data/sft_funcall/train.jsonl` |
| 5 — funcall SFT | base=**sft_code**（非旧 sft），4000 步 | `checkpoints/codechat_8b_funcall_v3/latest.pt` |
| 6 — RL（可选）过滤 | `scripts.filter_mbpp_by_passrate` | `data/mbpp_rl_curriculum.jsonl` |
| 7 — RL（可选）GRPO | base=**sft_code**，tiered + group=8 + log rollouts | `checkpoints/codechat_8b_rl_v3/latest.pt` |

### 3.4 关键设计决策

1. **RL base 用 sft_code，不用 funcall**
   funcall SFT 专门优化 `<functioncall> JSON` 格式输出，会把模型往结构化 JSON 拉；而 MBPP RL 的奖励完全基于可执行 Python，格式偏好和奖励目标正交甚至冲突。因此 RL 仍从代码能力最强的 `sft_code` 出发。

2. **funcall SFT 也从 sft_code 出发，而不是旧 sft**
   funcall SFT 在 sft_code 之上做，可以同时保留更强的代码能力 + 学会 function calling。只要 funcall 数据里 assistant 段不是清一色 JSON（实际它的 final 轮也是自然语言总结），代码能力退化有限。

3. **lr 2e-5（code SFT）< 3e-5（funcall SFT）< 5e-5（原始 SFT）**
   续训学习率阶梯式降低，避免后面的阶段破坏前面积累的分布对齐。

4. **GATE 打印而不强制中断**
   Stage 3 用内联 Python 计算 parseable/pass@k 并与阈值比较。不达标时只打 WARN，让用户可以基于诊断数据手动决定：是加大 `CODE_MAX_STEPS`、扩 `CODE_MAX_CF`/`CODE_MAX_STACK`，还是放弃 RL 先出 funcall ckpt。

### 3.5 超参与预算

| 阶段 | device_bs | grad_accum | world | global batch tokens | lr | steps | 预计耗时 |
|------|---|---|---|---|---|---|---|
| code SFT | 1 | 8 | 8 | 131k tok | 2e-5 | 2,000 | ~3.5h（按 v2 funcall 的 2000 步折算） |
| funcall SFT | 1 | 8 | 8 | 131k tok | 3e-5 | 4,000 | ~7h |
| pass@k 诊断 | — | — | 8 | — | — | 50 题 × 8 | ~1h |
| GRPO（可选） | — | — | 8 | — | 5e-6 | 500 | ~TBD |

总预算（不含 RL）：约 11~12 小时。

---

## 4. 期望结果 & 验证计划

### 4.1 Stage 2（code SFT）期望

- TB 上 `sft/loss` 应从 ~1.0 继续下降到 0.3 ~ 0.5（更结构化的代码数据 → loss 会比 funcall 的 0.6 更低）
- 不应出现大段 loss 尖刺（lr 已降到 2e-5）

### 4.2 Stage 3（诊断 GATE）期望

- **达标线**: parseable ≥ 50%、pass@8 ≥ 10%
- **期待值**: parseable 70% ~ 85%、pass@1 5% ~ 15%、pass@8 20% ~ 40%

如果达标：继续 Stage 4/5，并在 RL 环节观察 reward 曲线能否脱离恒 0。

如果 parseable ≥ 50% 但 pass@8 < 10%：模型会写 Python 但不会解 MBPP 题——考虑把 `CODE_MAX_CF` 再提到 40k，或追加一个小规模 instruction-tuning 轮次（HumanEval-Pack 的 synthesized 训练子集是备选）。

如果 parseable < 50%：数据或步数不够；优先加 `CODE_MAX_STEPS` 到 3000 ~ 4000，再检查 the-stack-smol 的 docstring 过滤是否太严格。

### 4.3 Stage 7（RL）期望

- `rl/reward` 曲线在前 50 步内离开 0（tiered reward 的最低档 0.05 应能在 group_size=8 下被触发）
- 每 50 步的 rollout 样本日志里应出现至少 1 个 parseable 的输出
- 500 步后预期 pass@1 上升 5 ~ 15 个百分点

---

## 5. 关键文件索引

```
runs/train_a800_x8_v2_funcall.sh     # v2 原始脚本（funcall + 失败的 RL）
runs/train_a800_x8_v3_funcall.sh     # v3 新脚本（code-boost + funcall + GATE + RL）
scripts/prepare_sft_code.py          # v3 新增：聚合 mbpp/codeforces/the-stack-smol
scripts/prepare_sft_funcall.py       # v2 复用
scripts/chat_sft.py                  # FSDP SFT，两阶段都用它
scripts/eval_mbpp_pass_at_k.py       # 诊断
scripts/filter_mbpp_by_passrate.py   # 过滤
scripts/chat_rl.py                   # GRPO

checkpoints/codechat_8b/latest.pt            # 预训练 (16GB)
checkpoints/codechat_8b_sft/latest.pt        # 通用 SFT（v2 base，v3 code-SFT 的起点）
checkpoints/codechat_8b_funcall/latest.pt    # v2 funcall（已产出）
checkpoints/codechat_8b_sft_code/latest.pt   # v3 新增（待跑）
checkpoints/codechat_8b_funcall_v3/latest.pt # v3 新增（待跑）
checkpoints/codechat_8b_rl_v3/latest.pt      # v3 新增（可选，RUN_RL=1 才有）

data/sft_code/train.jsonl           # v3 新增
data/sft_funcall/train.jsonl        # v2 产出，v3 复用
data/mbpp_passrate.jsonl            # Stage 3 产出
data/mbpp_rl_curriculum.jsonl       # Stage 6 产出（RUN_RL=1）

runs/tb/codechat_8b_funcall/        # v2 funcall TB
runs/tb/codechat_8b_sft_code/       # v3 code SFT TB（待产出）
runs/tb/codechat_8b_funcall_v3/     # v3 funcall TB（待产出）
runs/tb/codechat_8b_rl_v3/          # v3 RL TB（可选）
```

---

## 6. 启动命令

```bash
# 默认跑到 funcall SFT（含诊断 GATE 检查，不跑 RL）
bash runs/train_a800_x8_v3_funcall.sh

# 全跑，含 RL
RUN_RL=1 bash runs/train_a800_x8_v3_funcall.sh

# 已有 code ckpt，跑诊断 + funcall
SKIP_TO=3 bash runs/train_a800_x8_v3_funcall.sh

# 只跑 RL（已有 sft_code + mbpp_passrate.jsonl）
RUN_RL=1 SKIP_TO=6 bash runs/train_a800_x8_v3_funcall.sh

# 重建 code SFT 数据
FORCE_PREP_CODE=1 bash runs/train_a800_x8_v3_funcall.sh

# 扩容数据
CODE_MAX_CF=40000 CODE_MAX_STACK=30000 bash runs/train_a800_x8_v3_funcall.sh
```

---

## 7. 待办 & 风险

- [ ] `bigcode/the-stack-smol` 的 `data_dir="data/python"` 约定在新版 `datasets` 可能需要换成 `data_files=...`——首次运行若失败，fallback：`CODE_SOURCES=mbpp,codeforces` 先跑通，再补 the-stack。
- [ ] Codeforces-Python-Submissions 的字段名 (`problem-description`/`input-specification`/...) 依赖数据集当前 schema；换版本时需同步 `iter_codeforces`。
- [ ] GATE 阈值写死 parseable 50% / pass@8 10% 是经验拍脑袋；若首轮诊断 pass@8 达到 5% 就判断为能跑 RL，可以 `GATE_PASS_AT_K=0.05 bash ...` 直接覆盖。
- [ ] Stage 2 的 2000 步有可能不够。如果 loss 末段仍在 0.8 以上，说明还在强拟合过程中，建议先 SKIP_TO=2 再跑 2000 步（从 `codechat_8b_sft_code` 继续，lr 降到 1e-5）。
- [ ] RL Stage 7 里仍保留 `--sft-ckpt=sft_code`。如果实际观察到 funcall ckpt 的代码能力并未退化（可通过 `chat_cli` 手工 smoke），可以改为从 funcall ckpt 启动 RL，让最终模型一次性具备代码 + function calling 两种能力。

---

## 8. v3 实际运行复盘（2026-04-15）

### 8.1 运行概要

```
命令:        RUN_RL=1 bash runs/train_a800_x8_v3_funcall.sh
TB 日志:     runs/tb/codechat_8b_sft_code/          (code SFT, 2000 步)
             runs/tb/codechat_8b_funcall_v3/        (funcall SFT, 4000 步)
ckpt 产出:   checkpoints/codechat_8b_sft_code/latest.pt      ✅ 16G
             checkpoints/codechat_8b_funcall_v3/latest.pt    ✅ 16G
             checkpoints/codechat_8b_rl_v3/latest.pt         ❌ 未产出
```

### 8.2 阶段实际状态

| 阶段 | 预期 | 实际 |
|------|------|------|
| 1 — code SFT 数据 | 产出 `data/sft_code/train.jsonl` | ✅ DONE |
| 2 — code SFT 2000 步 | loss 降至 0.3 ~ 0.5 | ✅ DONE（具体 loss 曲线见 TB `codechat_8b_sft_code/`，待核对） |
| 3 — pass@k 诊断 GATE | parseable ≥ 50% 且 pass@8 ≥ 10% | ❌ **隐式 FAIL**（从 Stage 6 回推：pass_rate 分布仍两极化，GATE 必然未过） |
| 4 — funcall 数据准备 | 复用 v2 | ✅ DONE |
| 5 — funcall SFT 4000 步 | 基于更强 base，loss 应更低 | ⚠️ **完成但没提升**（见 8.3） |
| 6 — 按 pass rate 过滤 | 有题命中 `[0.05, 0.95]` | ❌ **FAIL**（`no problems fell in pass_rate`，fallback `mean_tiered >= 0.1` 也 0 题） |
| 7 — 优化版 GRPO | 500 步 reward 曲线离开 0 | ❌ **未启动**（与 v2 同样结局） |

### 8.3 关键证据 A：funcall SFT v3 与 v2 的 loss 曲线几乎完全重合

> **v3 的核心赌注是 "code-SFT 可以让后续 funcall SFT 起点更强"。下面这张对照表直接否决了这个假设。**

| step | v2 loss | v3 loss | Δ |
|---|---:|---:|---:|
| 20 | 2.1333 | 2.1573 | +0.024 |
| 100 | 0.6546 | 0.6679 | +0.013 |
| 200 | 0.5899 | 0.5885 | −0.001 |
| 1000 | ~0.6 | ~0.6 | ≈ 0 |
| 3500 | 0.5782 | 0.5811 | +0.003 |
| 4000 | 0.6291 | 0.6330 | +0.004 |

两条 loss 曲线的差异远小于随机种子噪声。**从优化动力学看，code-SFT ckpt 在 funcall 数据上的初始表现与旧 sft ckpt 无法区分**——2000 步、262M supervised token 的代码续训，对下游 funcall loss 的提升 < 1%。

耗时对比也吻合这个判断：
- v2 funcall: 25,482 s（7h 4min）
- v3 funcall: 25,444 s（7h 4min）

### 8.4 关键证据 B：Stage 6 过滤仍然 0 命中

脚本 stdout:
```
==> [6/7] filter MBPP by pass rate in [0.05, 0.95]
WARN: no problems fell in pass_rate in [0.05, 0.95]; falling back to mean_tiered >= 0.1
```

`mean_tiered >= 0.1` 是 v2 时代就写入 `scripts/filter_mbpp_by_passrate.py` 的 fallback——作用是 "哪怕 pass 率恒 0，只要 tiered reward 有均值也保留"。**fallback 之后依然没东西**，意味着在 sft_code ckpt 上连 tiered reward（能运行、能解析）的 0.1 均值都难达到。

推论：Stage 3 诊断的 MBPP 数字相对 v2 并没有质的变化，可能 parseable/pass@8 分布仍远低于 GATE 的 50%/10%（具体数字在 `data/mbpp_passrate.jsonl`，需要在训练机上 `cat data/mbpp_passrate.jsonl | jq ...` 核对）。

### 8.5 根因分析（为什么 code-SFT 没奏效）

按假设从强到弱排：

1. **分布不匹配是主因**：MBPP 要求 "短 NL prompt → 独立完整可被 `def foo(...)` 外调用的函数"。v3 的 code-SFT 三源里：
   - **the-stack-smol** 训的是 "module docstring → 整个 .py 文件"——风格偏工程代码（含 import/class/CLI），不是函数级；
   - **Codeforces-Python** 训的是 "题面 → `input()/print()` 解题脚本"——I/O 流派，MBPP 完全不用；
   - **MBPP 非 train 切分**（~600 条）数据量太小，淹没在 40k 条噪声里。
   结果模型"学会了写 Python 文件"，但**没学会 MBPP 期望的那种函数级短回答**。

2. **诊断 prompt 本身的格式与 SFT 训练格式未对齐**。`scripts/eval_mbpp_pass_at_k.py` 的 prompt 模板是固定写死的，若与 `prepare_sft_code.py` 里三源各自的模板有一处 token 不一致，就足以让模型输出跑偏。**需要 diff 两份模板**。

3. **2000 步 × lr 2e-5 仍不足**。TL;DR 给出的 262M supervised token，对 8B 从头预训练模型的 "新分布泛化"，可能远不够。v2 funcall 尚且跑 4000 步才勉强收敛到 0.6，code-SFT 同数量级数据跑一半步数本就偏保守。

4. **基座预训练的代码 token 占比太低**。v2 复盘 §2.1 已指出 `codechat_8b` 预训练只 3.9B token、代码占比未审计。"续训 2000 步" 救不回 "预训练根本没见过的分布"。

5. **MBPP 二值奖励在弱基座上天然无梯度**。即便把 parseable 提到 30%，只要 pass@8 还在 0 附近，`pass_rate ∈ [0.05, 0.95]` 的筛选条件仍然过不了。这是 v2 已经吃过的坑，v3 选择 "加强 base" 路线而不是 "改 reward 路线"——事实证明前者 cost 远大于后者。

### 8.6 v3 与 v5 的决策分水岭

v3 失败后没有再尝试 "code-SFT 再加 2000 步 / 换数据源 / 调 lr" 这类局部补丁（每次迭代成本约 11h × 8 卡），而是直接切到 v5：

| 维度 | v3（本报告） | v5（后续脚本） |
|------|------|------|
| 主线任务 | funcall SFT + **MBPP RL** | funcall SFT + **funcall RL** |
| SFT 与 RL 数据分布 | 不同（glaive vs MBPP） | **同分布**（都来自 glaive-function-calling-v2） |
| 奖励 | MBPP tiered（要求能运行 Python） | format-first 阶梯（只要求输出 `<functioncall>{JSON}` 格式） |
| RL 起飞的前置条件 | base 能写可解析 Python | base 能复现 `<functioncall>` tag（funcall SFT 后必然满足） |
| 每 rank 是否不同 prompt | 否（dist.broadcast 同一 prompt） | 是（每 rank 独立切片） |
| KL 正则 | 有 | 无（dense reward 自正则） |
| 参考思路 | 自研 tiered reward | 对齐 `reports/CodeChat_VS_MathGPT.md` 里 MathGPT 的可跑配方 |

本质上：v3 试图 "抬高 base 让旧 reward 有信号"，v5 换成 "换一个在当前 base 上就有信号的 reward"。从 cost/收益看，v5 是正确决策——**弱基座下改 reward 永远比加 SFT 数据更便宜**。

### 8.7 v3 有价值的副产物

虽然 RL 没跑起来，v3 并不是纯粹浪费：

- ✅ `checkpoints/codechat_8b_sft_code/latest.pt` 可作为代码能力基准 ckpt（即便没超过 GATE，smoke 起来应比 `codechat_8b_sft` 略好），后续做纯代码问答评测时可用。
- ✅ `checkpoints/codechat_8b_funcall_v3/latest.pt` **是当前最好的 funcall SFT ckpt**——和 v2 的 funcall ckpt 相比，虽然 loss 曲线无差异，但它是从 sft_code 出发的，理论上同时保留了 funcall 格式与部分代码能力。如果 v5 funcall SFT 来不及跑，这个 ckpt 可以作为 v5 RL 的起点。
- ✅ 验证了 "先做代码续训" 这条路径的投入产出比——阴性结果也是结论。

### 8.8 待核对项（下次登训练机时跑）

> 这些命令需要在训练机上执行，本地 context 看不到结果。

```bash
# 1) 确认 Stage 3 诊断的真实数字（是否 parseable 真的没上去）
jq -s '{
  n: length,
  parseable_any: [.[] | select((.parseable_rate // .parseable // 0) > 0)] | length,
  pass_any:      [.[] | select((.pass_rate // 0) > 0)] | length,
  mean_tiered:   ([.[] | (.mean_tiered // 0)] | add / length),
  mean_parseable:([.[] | (.parseable_rate // .parseable // 0)] | add / length)
}' data/mbpp_passrate.jsonl

# 2) 对比 code-SFT 两个 ckpt 的 loss 终值（TB 直出）
tensorboard --logdir runs/tb --bind_all
# 重点看: codechat_8b_sft_code/sft/loss 末段是否真的降到 < 0.5

# 3) sft_code vs sft 的 MBPP smoke（把两条 ckpt 都跑一遍 chat_cli 人肉对比）
python -m scripts.chat_cli --ckpt checkpoints/codechat_8b_sft/latest.pt
python -m scripts.chat_cli --ckpt checkpoints/codechat_8b_sft_code/latest.pt
# 输入相同的 MBPP prompt，看前者是否真的完全写不出函数、后者是否明显更好
```

### 8.9 给未来 v4/v6 的建议（如果仍想救 MBPP RL 路线）

仅在 v5 funcall RL 失败或不被产品采纳时启用：

1. **对齐 prompt 格式**：把 `scripts/eval_mbpp_pass_at_k.py` 的 prompt 模板与 `scripts/prepare_sft_code.py` 的 MBPP 分支对齐（包括特殊 token、换行数、`# Your code here:` 这种引导词）。这一步 0 成本但很可能直接把 parseable 从 1.75% 抬到 30%+。
2. **换 code-SFT 数据源**：用 `HumanEval-Pack`、`openai_humaneval`、`evol-instruct-code-80k`、`OpenCoder-SFT-Stage1` 这类 **函数级 instruction→code** 数据替代 Codeforces/the-stack-smol。数据量 10~30k 即可，比 40k 噪声有效。
3. **改 reward 而不是改 base**：即便停留在 v2 的弱 base，把 reward 从 "binary pass" 改成 "AST parseable + NameError 判定 + 至少一个 test 过的阶梯"，信号密度会上来。这其实就是 v5 在 funcall 任务上做的事情的 MBPP 翻译。
4. **GATE 写硬中断**：v3 的软 GATE（打 WARN 继续）浪费了 7h funcall SFT 的 GPU 时间。下次 GATE 未过应直接 `exit 1`，逼人工先修数据/步数。

---
