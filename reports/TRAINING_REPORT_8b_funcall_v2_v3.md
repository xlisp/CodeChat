# CodeChat 8B Function-Calling + MBPP RL 训练报告（v2 复盘 / v3 设计）

**日期**: 2026-04-14 ~ 2026-04-15
**硬件**: 8× A800-SXM4-80GB (NVLink)
**框架**: PyTorch 2.10 + FSDP (FULL_SHARD)，bf16
**相关脚本**: `runs/train_a800_x8_v2_funcall.sh`（v2）、`runs/train_a800_x8_v3_funcall.sh`（v3 新增）

---

## 0. TL;DR

- **v2 funcall SFT 阶段**: 成功完成 4000 步，loss 从 2.13 收敛至 0.5 ~ 1.0 区间，ckpt 保存到 `checkpoints/codechat_8b_funcall/latest.pt`。
- **v2 RL 诊断阶段**: 在 base ckpt `checkpoints/codechat_8b_sft` 上评测 50 题 × 8 采样 = 400 样本，**pass@1 = 0.00%、pass@8 = 0.00%、parseable = 1.75%、runnable = 1.25%**。过滤后无一题满足 `pass_rate ∈ [0.05, 0.95]`，RL 无法启动。
- **根因**: 不是 reward 设计或 group_size 的问题，而是基座模型的代码生成能力过低——连合法 Python 都写不出来。
- **v3 方案**: 在 funcall SFT 之前插入 “code SFT 续训” 阶段，数据扩展为 MBPP（非 train 切分）+ Codeforces-Python + the-stack-smol（docstring 子集）。诊断改为评测 **sft_code ckpt**，GATE 阈值 parseable ≥ 50% 且 pass@8 ≥ 10%，达标才进 RL。

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
