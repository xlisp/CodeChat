# CodeChat 8B 训练瓶颈诊断与路线图（post-v7）

**日期**: 2026-06-10
**性质**: 战略文档 —— 回答三个问题：(1) 如何训出更强的代码能力；(2) 如何避免灾难性遗忘；(3) 如何把 code / funcall / math 多种能力组合训练到同一个模型里。
**前置阅读**:
- `reports/TRAINING_REPORT_8b_a88_x8.md`（v1：预训练 + 通用 SFT + MBPP RL 失败）
- `reports/TRAINING_REPORT_8b_funcall_v2_v3.md`（v2/v3：MBPP RL 抢救失败）
- `reports/TRAINING_REPORT_8b_funcall_v5.md`（v5：funcall RL 成功但成专家）
- `reports/TRAINING_REPORT_8b_v6_unified.md`（v6：joint SFT，code 被覆写）
- `docs/codechat_8b_v7_design.md`（v7：已设计、已写码、**尚未跑**）
- `reports/CodeChat_VS_MathGPT.md`（为什么 MBPP RL 无信号而 GSM8K RL 有）
- `docs/mixed_sft_vs_moe.md`（为什么多能力用混合 SFT 而不是 MoE）

---

## 0. TL;DR

1. **v7 已就绪未跑，下一步就是跑它**。它修复的是 v6 的"遗忘"问题（token 失衡、判别捷径、监控缺失），不需要再改设计。
2. **但 v7 的天花板不在 SFT 层，在预训练层**。8.3B 参数只喂了 3.93B tokens（0.47 token/参数，Chinchilla 最优的 2.4%），且语料是**纯 Python 代码**——没有自然语言，没有数学。MBPP parseable 1.75% 的根因在这里。SFT 配比调得再完美，也只能让模型"不忘记它本来就很弱的代码能力"。
3. **提升代码能力上限的唯一解是续预训练（mid-training）**：再喂 30B+ tokens 的 code+NL+math 混合语料，最后带一个高质量退火阶段。约 2 周 ×8 卡。
4. **灾难性遗忘有一套可复用的防治规则**（§4），核心是：按监督 token 配比、任何续训带 replay、从中性 base 出发不串专家 ckpt、控制 epoch、消除捷径特征、分域监控。
5. **加数学 = v7 框架加第四个 `source`**，但前置条件是 mid-training 先把数学语料喂进 base，否则重演 MBPP 的故事（base 无先验 → SFT 只学到格式 → RL 无信号）。数学是 RL 信号最便宜的域（MathGPT 同框架已验证），做完数学 SFT 后 RL 第一次能真正赚到能力。

---

## 1. 现状盘点：v1 → v7 我们学到了什么

| 版本 | 做了什么 | 结果 | 教训 |
|---|---|---|---|
| v1 | 3.93B tokens 预训练 + 38k Alpaca SFT + MBPP binary-reward GRPO | RL reward ≡ 0.000 跑了 415 步 | 弱 base 上二元 reward 无梯度；RL 不能修复 SFT 没教会的能力 |
| v2 | tiered reward + pass-rate 过滤抢救 MBPP RL | 仍无信号（pass@1 = 0.00%） | 当 group 内全 0 时，阶梯放大的是噪声不是信号 |
| v3 | 先补 code SFT 再试 RL | funcall loss 曲线与 v2 重合，无收益 | parseable 1.75% 的 base，SFT 续训救不动 MBPP |
| v5 | 复刻 MathGPT recipe：SFT 与 RL 同分布（glaive funcall） | funcall pass@1 86.4%，RL 收益 <1pp 后饱和 | "选 SFT 已会做的任务做 RL" 是对的；但 RL 从 85% 起点只能空转 |
| v5 副作用 | — | 模型变 funcall 专家，"write quicksort" 也 emit `<functioncall>` | 在多能力 ckpt 上跑单域数据 = 灾难性遗忘 |
| v6 | joint SFT（code+funcall 混合 shuffle） | funcall 保住 85.4%，**code 输出乱码** | 行数比例 5.6:1 骗人，监督 token 比例实际 ~30:1；total loss 掩盖单域退化 |
| v7 | token 平衡 + 20% system 注入 + 10% 判别负例 + 分域监控 | **未运行** | （待验证） |

五个版本反复验证的三条铁律：

1. **RL 只能放大 SFT 已有的能力，不能无中生有**。RL 起点 pass@1 < 1% → 无信号；> 85% → 饱和。RL 的甜区是 5%–60%。
2. **CE loss 是 token 级累积量**，哪个域的监督 token 多，参数就被拉向哪个域。配比必须按 token 算。
3. **模型会学到最便宜的判别函数**。v6 的 code 样本从不带 `<|system|>`，funcall 样本必带，于是模型学到的是 "看见 system tag 就调工具" 而不是 "理解用户意图"。

---

## 2. 瓶颈分层诊断

### 2.1 第一层（根本）：预训练严重欠训练 + 语料单一

**欠训练的量化**：

| 模型 | 参数量 | 预训练 tokens | token/参数比 |
|---|---|---|---|
| Chinchilla 最优 | — | — | ~20 |
| LLaMA-7B | 6.7B | 1.0T | ~149 |
| GPT-3 | 175B | 300B | ~1.7 |
| nanochat d20 | 561M | ~11B | ~20 |
| **CodeChat 8B** | **8.3B** | **3.93B** | **0.47** |

CodeChat 8B 的预训练量是 Chinchilla 最优（~166B tokens）的 **2.4%**。预训练 loss 0.609 看起来不错，但那是在纯代码语料上——代码高度模板化，loss 天然低，不代表能力。v1 报告 §6 自己的结论也是"容量不足……远达不到能稳定写出可运行代码"，建议 9 是"换更强的 base"。这个判断至今成立，只是被 v5/v6 的 funcall 路线暂时绕开了（funcall 是格式学习，对 base 要求低；写对能跑的代码是语义任务，对 base 要求高）。

**语料单一的后果**：

预训练数据是 `codeparrot/github-code-clean` 的 Python 子集（32 shards ≈ 4.2B tokens，30k 步基本过了一遍）。这意味着：

- **没有自然语言** → 模型读不懂 MBPP 的英文题面。MBPP 的输入是 NL 描述，输出才是代码；NL 理解是空中楼阁，写码无从谈起。这是 parseable 1.75% 的另一半根因（第一半是欠训练）。
- **没有数学** → 未来加 GSM8K SFT 时，模型对 "Janet has 3 apples..." 这类文本的先验接近零，SFT 只能学到 `#### N` 的格式外壳，学不到推理。会重演 MBPP 的故事。
- glaive funcall 能学到 86% 是因为它**恰好是格式任务**：JSON schema → JSON call，模板性极强，弱 base 也能背下来。这反而掩盖了 base 的真实水平。

### 2.2 第二层：SFT 数据工程（v6 暴露，v7 已修，待验证）

详见 `docs/codechat_8b_v7_design.md` §1。三个独立问题：

1. **监督 token 失衡 ~30:1**（funcall 平均行长 ~7,500 token vs code ~1,500 token，行数再差 5.6 倍）。
2. **判别捷径**：`<|system|>` 的有无成了完美但错误的判别函数。
3. **监控缺失**：14h 训练只看 total loss，code 退化到 18.5h 后的 smoke test 才发现。

v7 的修复（token 平衡到 ~1:1.2、20% system 注入 + 10% 负例、每 200 步分域 held-out loss + 每 500 步 binary smoke）是正确的，**本文档不改动 v7 设计**。

### 2.3 第三层：RL 策略

- **funcall RL 已饱和**：v5/v6 两次证明从 85% 起点 REINFORCE 收益 <1pp。eval 集 122 题已被做穿（pass@16 = 0.94）。继续投入需要更难的 eval（§6.4）。
- **MBPP RL 被 VERDICT 门控**：`eval_mbpp_pass_at_k` 的三档判据（<1% 别跑 / 1-5% tiered+filter / ≥5% 标准 GRPO）是对的，v7 训完后重测一次决定分叉。
- **数学 RL 是未开发的甜区**：MathGPT 在 560M 模型上用 `#### N` 正则匹配的二元 reward 从 0.41 起步跑到 Pass@1 14.75%。同样的框架、更大的模型、前提是 SFT 先把格式教会 + base 里有数学先验。

### 2.4 瓶颈层级总表

| 层 | 问题 | 状态 | 对应行动 |
|---|---|---|---|
| 预训练 | 0.47 token/参数；语料纯 Python 无 NL/数学 | **未解决，真正的上限** | §5 mid-training |
| SFT 数据 | token 失衡 + 判别捷径 | v7 已修，待验证 | §3 跑 v7 |
| SFT 监控 | 只看 total loss | v7 已修 | §3 |
| RL | funcall 饱和；MBPP 无信号；math 未做 | 已认清 | §6 |

---

## 3. 第一步：跑 v7 并验收（~11h，本周可完成）

v7 的设计、监控指标、退化模式判别已写在 `docs/codechat_8b_v7_design.md` §4，此处只补执行层面的三点：

### 3.1 跑前检查

```bash
bash runs/train_a800_x8_v7.sh          # stage 1 会打印 token 比例估算
```

Stage 1 结束时确认打印的 **code+negative : funcall 监督 token 比例落在 0.6–1.6**（脚本超出会 WARN）。v6 最大的坑就是只看了行数比例。如果偏出区间，用 `PREP_FUNCALL_CAP` / `PREP_MAX_*` 调整后 `FORCE_PREP=1` 重做。

### 3.2 训中监控（kill 判据）

TensorBoard 盯四条线：`sft/loss_code`、`sft/loss_funcall`、`sft/smoke_code_pass`、`sft/smoke_funcall_pass`。任何一个域的 held-out loss **连续 3 个 eval 点（600 步）单调上涨**，立即 kill —— 这就是 v7 花成本做分域监控的全部意义：v6 浪费的 18.5h，v7 应该在 1-2h 内止损。三种退化模式的修复 knob 见 v7 设计文档 §4.2（`PREP_FUNCALL_CAP` / `PREP_NEG_FRAC` / `SFT_SYSTEM_INJECT_RATIO`）。

### 3.3 训后验收：两个 eval 决定后续路线

```bash
# (a) funcall 不回退验证
.venv_train/bin/python -m scripts.eval_funcall \
    --ckpt checkpoints/codechat_8b_sft_v7/latest.pt --num-samples 16

# (b) code 路线分叉判据
.venv_train/bin/python -m scripts.eval_mbpp_pass_at_k \
    --ckpt checkpoints/codechat_8b_sft_v7/latest.pt --k 8
```

| Eval | 期望 | 含义 |
|---|---|---|
| funcall pass@1 | ≥ 0.80 | 容忍比 v6 的 0.854 略降——这是把 funcall 曝光量从 84.8% 砍到 ~45% 的合理代价。低于 0.75 说明 cap 砍太狠，升 `PREP_FUNCALL_CAP` 重训 |
| MBPP VERDICT | ≥ 1%（理想 ≥5%） | **分叉点**：≥5% → code GRPO 第一次有戏（§6.3）；1–5% → tiered+filter 可试；<1% → SFT 层确认无解，§5 mid-training 升为最高优先级 |

注意：v7 的 funcall smoke 只检查 `location` 字段**存在性**，不查值。v6 的 "Weather in Tokyo? → Berlin" 参数幻觉在 v7 大概率仍在（设计文档 §7.1 已自认），修法在 §6.4，不阻塞 v7 验收。

---

## 4. 灾难性遗忘防治手册

这一节把 v5/v6/v7 的教训提炼成**以后每加一个能力域都照此执行**的规则。先讲机制，再列规则。

### 4.1 机制：遗忘不是"渐忘"，是"覆写"

SGD 没有"保护旧知识"的概念。CE loss 对参数的拉力正比于**监督 token 的累积量**：v6 中 funcall 累积 ~847M 监督 token，code 只有 ~30M，参数被以 28:1 的力量拉向 funcall 分布。code 能力不是慢慢淡忘的，是被数量级压制直接覆写的——这就是为什么 8000 步训完，"write quicksort" 输出的不是"差一点的代码"而是乱码：那部分参数已经被征用了。

推论：**防遗忘的本质是控制各域监督 token 的梯度累积量之比**，所有规则都从这里推出来。

### 4.2 规则清单

**规则 1 — 按监督 token 配比，永远不按行数。**
行数比例 5.6:1 在 token 层面是 30:1。配比目标：各域监督 token 量 1:1（按域难度可微调到 1:1.5 以内）。`prepare_sft_v7.py` 的 `len(content)/3.5` 粗估（GPT-2 BPE 在 Python 上 ~3.5 chars/token）误差 10-20%，足够指导 cap。

**规则 2 — 任何续训阶段都带 replay。**
在已有多能力的 ckpt 上做任何续训（新域 SFT、RL、领域适配），必须混入 **10–30% 旧域数据**。v5 的事故就是在通用 SFT ckpt 上跑了 6000 步纯 funcall：funcall 专家 prior 强到 86% rollouts 无条件 emit `<functioncall>`。RL 阶段同理——rollout prompt 池单一域时，policy 会整体漂向该域（§6.2 的混合锚定）。

**规则 3 — 从中性 base 出发，不串专家 ckpt。**
v7 设计 §3 "不从 v6 续训" 的理由要固化成惯例：specialist ckpt 的分布偏移做逆向工程，成本高于从中性 base 重训（v6 报告 §6.1 方案 C 的"双向拉锯"风险）。能力栈的正确叠法是**回到分叉点重训**，不是在错误的 ckpt 上打补丁。

**规则 4 — 控制 epoch 数，混合 SFT 以 2–3 epochs 为上限。**
v6 跑了 ~7 epochs，step 3000 后 loss 不再下降，后 5000 步全在过拟合 funcall 顺便覆写 code（单步 loss 低至 0.065 = memorization 信号）。数据量决定步数，不是反过来：`max_steps ≈ 总监督 token × 2.5 / 131k`。

**规则 5 — 消除捷径特征（shortcut audit）。**
每加一个新域，问一个问题：**"模型能否用一个表面 token 模式（而非语义）判别这个域？"** 能，就必须打破：

| 捷径类型 | v6 实例 | 打破手段 |
|---|---|---|
| 标签存在性 | `<|system|>` 有无 ↔ funcall/code | 20% system 注入（让标签在所有域出现） |
| 标签内容 | system 里有 tool schema ↔ 必须调用 | 10% 判别负例（有 schema 但答案是 code） |
| 长度/格式 | （潜在）数学题全是短 user 问题 | 混入长 context 数学样本 |
| 语言 | （潜在）某域全英文/全中文 | 双语覆盖或显式接受该捷径 |

注入和负例必须**配对**（v7 设计 §2.2 的论证）：只注入 → 没见过"有 schema 但不该调"；只负例 → 标签仍 90% 与 funcall 共现。

**规则 6 — 每域一份 held-out loss + 一个 binary smoke，训练中实时监控。**
loss 是连续信号看 trend，smoke 是离散信号看"会不会做"，两层互为确认（v7 设计 §2.3）。新加域 = 新加一份 `eval_<domain>.jsonl`（300 行，shuffle 前切出）+ 一个 smoke prompt + 两条 TB 曲线。**没有分域监控的多域训练等于盲飞**——v6 的 total loss 全程漂亮地下降，code 在水面下死掉。

**规则 7 — 续训用更低的 lr 和更短的 schedule。**
能力保持类续训（如在 v7 ckpt 上补一个域）lr 用主 SFT 的 1/3（1e-5 vs 3e-5），步数 2000-3000，且按规则 2 带 replay。lr 越高、步数越长，旧分布被覆写越快。

**规则 8 — 参数层手段是储备，不是首选。**
KL-to-base 正则、checkpoint 权重平均（model soup / WiSE-FT）、LoRA-per-domain 再合并，这些都能缓解遗忘，但：(a) 数据层手段（规则 1-7）没榨干之前上它们是过度工程；(b) 各自有代价——KL 正则要存 ref model（8B fp32 又是 32GB host RAM），LoRA 合并的能力干涉不可控。**触发条件**：如果 v7 验收通过但后续加第 3、4 个域时 token 平衡 + 注入 + 负例仍压不住互相干涉，再考虑 LoRA-per-domain；如果域数超过 ~5 个，按 `docs/mixed_sft_vs_moe.md` §5 的判据重新评估 MoE。

---

## 5. 第二步：mid-training —— 提升 code 能力上限的唯一解

这是本路线图中**唯一能抬高天花板**的投资，其余动作都是在现有天花板下优化。

### 5.1 为什么是续预训练而不是更多 SFT

SFT 数据（千~十万行级）教的是**格式和行为**；语言理解、代码语义、数学先验这种**知识性能力**来自预训练的 token 量。v3 已经做过实验：在 v1 base 上补 20k 行 code SFT，MBPP 毫无起色。1.75% parseable 的问题不在"没见过题型"，在"底子里没有足够的 代码↔自然语言 对应关系"。

### 5.2 数据混合设计

从 `checkpoints/codechat_8b/latest.pt` 续训（预训练本就不存 optimizer state，restart 优化器是设计内行为，warmup 重走即可）。目标 **30B tokens**（最低有效剂量；预算充足可到 80B）：

| 来源 | 占比 | tokens | 作用 |
|---|---|---|---|
| Python 代码（the-stack-v2 dedup Python 子集） | 45% | 13.5B | **replay + 扩容**——规则 2 在预训练层的应用：纯新分布续训会反过来遗忘已有的 code prior |
| 自然语言（FineWeb-Edu 或同级过滤 web 文本） | 30% | 9B | 修 NL 理解——MBPP 题面、指令跟随、对话能力的地基 |
| 数学（OpenWebMath + proof-pile-2 的 algebraic-stack） | 15% | 4.5B | 为 v8 数学能力埋先验（§7.2 的前置条件） |
| 代码↔NL 对照（StarCoder 的 jupyter-structured、文档字符串丰富的代码） | 10% | 3B | 直接练 "NL 描述 → 代码" 的映射，对 MBPP 最对症 |

**实现**：写 `scripts/prepare_pretrain_v2.py`，按比例 round-robin 从各 HF 数据源取文档、统一 GPT-2 BPE tokenize、写混合 shards 到 `data/pretrain_v2/`。混合在**准备期**完成，`PretrainLoader` 一行不用改（符合 CLAUDE.md 的 simplicity-first；多源加权采样 loader 是不必要的抽象）。

### 5.3 训练计划

| 阶段 | tokens | 步数 | lr | 数据 |
|---|---|---|---|---|
| 主段 | 27B | ~206k 步 | 1e-4 cosine，warmup 1000 | `data/pretrain_v2/`（§5.2 比例） |
| **退火段** | 3B | ~23k 步 | 从主段末端 lr 线性降到 1e-6 | 高质量子集：教科书式代码、MBPP/HumanEval 风格"NL 描述+函数+测试"文本、精选数学 step-by-step 解答 |

退火段（参考 Llama 3 / MiniCPM 的实践）是性价比最高的部分：在低 lr 下用高质量窄分布数据收尾，对下游 benchmark 的杠杆远超同量随机数据。退火数据单独准备成 `data/pretrain_anneal/`，第二次 `base_train` 调用从主段 ckpt 续。

**成本**：实测吞吐 22.3 Ktok/s（v1 报告）→ 30B tokens ≈ **374h ≈ 15.6 天** ×8 卡。这是整个路线图最大的单笔投资，但对照：v1-v6 已花费的 SFT/RL 实验加起来 ~60h，全部在 2.4% Chinchilla 的 base 上空转优化。

**降档选项**：预算紧张先做 10B tokens（~5.2 天），比例不变。10B 已能显著改善 NL 理解（nanochat 561M 用 11B tokens 达到可用对话水平），但代码语义提升幅度会小一些。

### 5.4 验收

mid-training 结束后、重做 SFT 之前：

1. `eval_mbpp_pass_at_k --ckpt checkpoints/codechat_8b_v2/latest.pt --k 8` —— base（未 SFT）就该看到 parseable 从 1.75% 显著上移；
2. 各域 held-out perplexity：code / NL / math 三份固定文本的 loss，与旧 base 对比，code loss 不得回退（replay 配比的验收）；
3. 在新 base 上重跑 v7 SFT（同配置、新 `--run-name`，例如 `codechat_8b_sft_v8`），对比 v7：MBPP pass@1、funcall pass@1、smoke 全套。

### 5.5 战略备选：收缩到 2–3B

同等算力下 8B×4B tokens 远不如 2B×30B tokens（Chinchilla）。2B 路线还能甩掉 FSDP 的全部复杂度（v5 的两个 NCCL 去同步 bug、save 时 32GB host RAM 压力都是 FSDP 税）。**不推荐现在切换**的理由：8B 管线已跑通、funcall 86% 证明管线能学会东西、mid-training 30B 后 8B 的 token/参数比到 4.1 仍欠训练但已脱离"严重"区间。但如果 mid-training 后 MBPP 仍 <5%，2B-dense×80B tokens 应作为 v9 的认真选项，届时 `--preset 2b` 单卡可训，迭代速度快 8 倍。

---

## 6. RL 策略：每个域分开判断

### 6.1 总原则（五个版本换来的）

RL 的甜区是 **SFT 起点 pass@1 ∈ [5%, 60%]**：低于下界无信号（v1/v2 MBPP），高于上界饱和空转（v5/v6 funcall）。每次 RL 前必须跑 pre-RL 诊断（lr=0 跑一次 eval），这个习惯已在管线里，保持。

### 6.2 RL 阶段的防遗忘

RL 也是续训，规则 2 适用。v5 的教训：纯 funcall prompt 池跑 RL，policy 整体漂向"万物皆 functioncall"。修法（按成本排序）：

1. **最便宜**：RL 步数压到甜区内的最小值（v7 已做：`RL_MAX_STEPS=60`），漂移与步数成正比；
2. **混合 prompt 池**：rollout prompt 中混入 20-30% 其他域的 prompt，reward 用对应域的 reward fn（code 域用 `execution.py`，math 域用 `#### N` 匹配）——这要求 `chat_rl_funcall.py` 泛化成多 reward 路由，是 v8 的代码改动项；
3. **RL 中加分域 smoke**：把 v7 的 smoke 机制搬进 RL loop，每 20 步跑一次全域 smoke，任何域 FAIL 即停。

### 6.3 Code RL（被 §3.3 的 VERDICT 门控）

- VERDICT ≥5%：标准 GRPO，binary/fractional reward 皆可，先跑 `filter_mbpp_by_passrate` 把 pass_rate ∈ [0.05, 0.95] 的题筛出来（全对和全错的题都无 advantage）；
- 1–5%：tiered reward + group_size ≥ 8 + 过滤（v2 的配方，当时失败是因为起点 0.00%，不是配方错）；
- <1%：不跑，等 mid-training。
- 更远期（base 足够强之后）：`docs/swebench_rl.md` 的 syntax → apply-only → docker 三级课程是 code RL 的下一形态，但它对 base 的要求比 MBPP 更高，在 MBPP pass@1 稳定 >20% 之前不要碰。

### 6.4 Funcall：RL 停止加注，转向数据修复 + 换难 eval

当前 122 题 eval pass@16 = 0.94，已无区分度。两个具体动作：

**(a) 参数幻觉修复（SFT 数据层，v8 随手做）**：glaive 高频参数值（Berlin/Paris/New York）被模型背死，问 Tokyo 答 Berlin。修法是 `prepare_sft_v7.py` 加参数值替换增强：检测 assistant 的 functioncall 参数值如果在 user query 中出现过，50% 概率把两处**同步**替换成同类型随机值（城市换城市、日期换日期）。强迫模型学"从 query 抄参数"而不是"输出高频值"。同时给 `funcall_reward.py` 加一档：参数值与 query 的字符串包含性检查，把 "格式全对但值是幻觉" 从 full_match (1.00) 降到 0.7。

**(b) 难化 eval（BFCL 风格）**：新 eval 集三类题各 1/3 —— 多工具选择（system 挂 3-5 个 schema 选对一个）、**无关工具拒绝**（挂着 schema 但 user 问题不该调用，正确行为是直接回答——v7 的判别负例天然是这类训练数据，eval 里也要有）、多轮调用（第一轮 response 后基于 function_response 决定第二轮）。glaive 原始数据里三类都有素材，`prepare_rl_funcall.py` 改抽取逻辑即可。

### 6.5 Math RL（v8 的主菜，见 §7.2）

唯一一个预期能从 RL 拿到大幅能力增益的域：reward 函数 10 行正则（MathGPT `tasks/gsm8k.py`），二元信号干净，GSM8K 7.5k 题量充足，且 MathGPT 已在 560M 模型上验证从 SFT 起点 ~40% 提升到 Pass@1 14.75%→ 等量级收益。前提链：mid-training 数学语料（§5.2）→ math SFT（§7.2）→ pre-RL 诊断落在甜区 → RL。

---

## 7. 多能力组合训练框架（v8+）

### 7.1 加一个新域的标准作业程序（SOP）

v7 的 `source` 字段 + 统一 conversations 格式就是多域框架。**每加一个域，走完这张 checklist**：

```
[ ] 1. 数据源选定，转成 conversations 格式，打 source 标签
[ ] 2. 监督 token 量估算，与现有各域 cap 到 ~1:1（规则 1）
[ ] 3. 切 300 行 held-out eval shard（shuffle 之前切！）
[ ] 4. 设计 1 个 binary smoke prompt + PASS 正则
[ ] 5. 捷径审计（规则 5）：新域有没有表面特征可被判别？
       需要的话扩展注入池 / 加跨域负例
[ ] 6. 重新 shuffle 全量 train.jsonl，从中性 base 重训（规则 3）
[ ] 7. 训中盯 N+1 组分域曲线；训后跑全域 eval 矩阵（§7.3）
[ ] 8. RL 仅在该域 pre-RL 诊断落在 [5%, 60%] 时启动（§6.1）
```

域数 2-5 个之内这套框架够用；超过 5 个或出现压不住的能力互斥，再按 `docs/mixed_sft_vs_moe.md` §5 评估 MoE / LoRA-per-domain（规则 8）。

### 7.2 数学域接入（v8 具体方案）

**前置条件（硬性）**：§5 mid-training 已完成且数学语料占比 ≥10%。在纯代码 base 上直接做数学 SFT 必然重演 MBPP：格式学得会（`#### N` 比 JSON 还简单），推理学不会，RL 起点趴在 <1%。

**SFT 数据**：

| 来源 | 量 | 处理 |
|---|---|---|
| GSM8K train | 7.5k | 原始 CoT 答案，结尾 `#### N` 格式保留 |
| MetaMathQA | cap 到与 code/funcall token 量对齐（约 30-40k 行） | 增广数据，过滤掉与 GSM8K test 重叠的题 |

三域 token 配比目标 **code : funcall : math ≈ 1 : 1 : 1**。数学样本平均较短（~500 token），行数 cap 会比 funcall 高，再次强调按 token 算。

**捷径审计（checklist 第 5 项）**：数学样本若全是"短 user 问题、无 system、英文"，模型可能用长度+语言判别。对策：数学样本同样吃 20% system 注入（注入池加 "You are a helpful math tutor" 类条目）；可选加少量 "system 挂 tool schema + user 问数学题 + assistant 直接算" 的跨域负例（与 v7 code 负例同构）。

**监控**：`eval_math.jsonl`（300 行）+ smoke prompt：

```
<|user|>\nNatalia sold clips to 48 of her friends in April, and then she sold
half as many clips in May. How many clips did Natalia sell altogether?\n<|end|>\n<|assistant|>\n
# PASS: 输出含 "#### 72"
```

**Math RL**（SFT 后，诊断在甜区时）：复用 `chat_rl_funcall.py` 的 REINFORCE-with-baseline 骨架，reward fn 换成 GSM8K 正则匹配（新文件 `codechat/math_reward.py`，~20 行）。超参起点照搬 v5：num-samples 16、lr 1e-5 × init_frac 0.05、每 30 步 eval。与 funcall RL 的关键差异：起点预期在 10-40% 而不是 85%，**这次 advantage 不会是 0**，预期能看到 MathGPT 式的真实爬升曲线。按 §6.2 规则混入 code/funcall smoke 防漂移。

### 7.3 全域 eval 矩阵（每个 ckpt 出厂前必跑）

能力组合训练的终极验收不是单域指标，是**矩阵无回退**：

| ckpt ↓ / eval → | MBPP pass@1 | funcall pass@1 | GSM8K pass@1 | smoke code | smoke funcall | smoke math |
|---|---|---|---|---|---|---|
| v7 sft | 基线 | 基线 | —（无先验） | ✓? | ✓? | — |
| v8 sft（mid-train 后） | **必须 > v7** | ≥ v7−2pp | 新基线 | ✓ | ✓ | ✓ |
| v8 math-RL | ≥ v8 sft−1pp | ≥ v8 sft−1pp | **必须 > v8 sft** | ✓ | ✓ | ✓ |

判定规则：**任何一格相对上一行回退超过 2pp，该 ckpt 不出厂**，回到对应层的 knob（配比 / replay / RL 步数）修复。把这张表写进 v8 的 pipeline 末尾 stage，自动打印。

### 7.4 v8 流水线草图

```
checkpoints/codechat_8b/latest.pt
        ↓ [v8-1] mid-training 主段 27B tokens (~13 天)        prepare_pretrain_v2.py / base_train.py
        ↓ [v8-2] 退火段 3B tokens (~1.5 天)                   data/pretrain_anneal/
checkpoints/codechat_8b_v2/latest.pt          ← 新的中性 base，跑 §5.4 验收
        ↓ [v8-3] 通用 SFT（38k Alpaca，同 v1 配置）
checkpoints/codechat_8b_v2_sft/latest.pt
        ↓ [v8-4] 三域 joint SFT（v7 配方 + math source，token 1:1:1，~6000 步）
checkpoints/codechat_8b_sft_v8/latest.pt      ← 跑 §7.3 矩阵
        ↓ [v8-5] math RL（诊断在甜区时；带跨域 smoke 哨兵）
checkpoints/codechat_8b_rl_v8/latest.pt       ← 再跑一次矩阵
        ↓ [v8-6] (VERDICT≥5% 时) code GRPO
```

照 CLAUDE.md 惯例：新脚本 `runs/train_a800_x8_v8.sh`，append-only，不改 v1-v7。

---

## 8. 优先级与时间线

| # | 动作 | 成本 | 解锁什么 | 依赖 |
|---|---|---|---|---|
| 1 | 跑 v7 + §3.3 验收 | ~11h | 防遗忘配方闭环验证；MBPP VERDICT 分叉 | 无 |
| 2 | funcall 参数增强 + 难化 eval（§6.4） | ~0.5 天（人力） | 参数幻觉修复进 v8 数据 | 无，可与 1 并行 |
| 3 | `prepare_pretrain_v2.py` + 语料下载 | ~1-2 天（人力+下载） | mid-training 就绪 | 无，可与 1 并行 |
| 4 | **mid-training 30B + 退火**（§5） | **~15.6 天 ×8 卡** | code/math 能力上限；v8 全部前置 | 3 |
| 5 | v8 三域 joint SFT（§7.2/7.4） | ~11h | 三能力统一 ckpt | 4 |
| 6 | math RL | ~5h | 第一次有真实梯度的 RL 增益 | 5 |
| 7 | (条件触发) code GRPO | ~10h | MBPP 实际提升 | 5 + VERDICT≥5% |

**关键路径是 #4**。#1/#2/#3 都是它的前菜或并行项；#4 不做，#5-7 全部在 2.4% Chinchilla 的地基上重复 v1-v6 的天花板。

---

## 9. 风险与备选

| 风险 | 概率 | 缓解 |
|---|---|---|
| v7 仍出现 code 退化（模式 A） | 中 | 设计文档 §4.2 的 knob 阶梯：`PREP_FUNCALL_CAP=20000` → `PREP_NEG_FRAC=0.20`；分域监控保证 1-2h 内发现 |
| mid-training 后 MBPP 仍 <5% | 中低 | 升级到 80B tokens 退火加重；或触发 §5.5 的 2B×80B 战略收缩 |
| mid-training 中 code perplexity 回退（NL 语料挤占） | 低 | 45% code replay 就是为此设计；训中加三域 perplexity 监控（v7 监控思想上移到预训练层） |
| GPT-2 BPE 对数字切分不利于数学 | 确定但可接受 | GSM8K 量级的算术 MathGPT 同类 tokenizer 已验证可行；**不换 tokenizer**——换词表会作废全部 ckpt，得不偿失 |
| 三域互斥压不住（v8 矩阵反复回退） | 低 | 规则 8 的储备手段：LoRA-per-domain → 合并；域数继续涨则评估 MoE |
| 8 卡被占 / 15 天窗口排不出 | — | 降档 10B tokens（5.2 天）先验证方向，结论成立再追加 |

---

## 10. 一句话总结

**短期跑 v7 验证防遗忘配方；中期用 mid-training（code+NL+math 混合续预训练 + 退火）抬高能力上限——这是"更强代码能力"的唯一解；此后每加一个能力域都走同一条已被 v1-v7 验证的路径：mid-train 埋先验 → token 均衡 joint SFT + 捷径打破 + 分域监控 → 仅在 pre-RL 诊断落在 [5%, 60%] 甜区的域上做 RL。**
