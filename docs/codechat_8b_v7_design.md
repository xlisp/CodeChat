# CodeChat 8B v7 训练设计文档

**日期**: 2026-04-26
**目标 ckpt**: `checkpoints/codechat_8b_sft_v7/latest.pt`
**承接**: v6 失败（`reports/TRAINING_REPORT_8b_v6_unified.md`）—— 联合 SFT 让 funcall pass@1 维持 85%，但 code 能力被完全覆写（"write quicksort" 输出乱码）。
**参考**: minimind 的统一 conversations 数据格式 + 20% 概率系统提示注入。

---

## 0. TL;DR

v7 是 v6 的针对性修复，三个核心改动一一对应 v6 的三个失败点：

| v6 失败点 | v7 修复 |
|---|---|
| 数据**严重失衡**：funcall:code ≈ 30:1（监督 token 量） | **Token 平衡**：cap glaive 至 30k 行，扩充 code 源到 ~140k 行，目标比例 ~1:1.2 |
| **判别信号缺失**：模型学到 "看到 `<\|system\|>` 就 emit `<functioncall>`"（因为 v6 的 code 样本从来没见过 `<\|system\|>`）| **20% system 注入** + **10% 判别性负例**：让 `<\|system\|>` 在普通对话中也常见，让 system+tools 同时出现但答案是 code 的样本也存在 |
| **监控缺失**：14h 训练只看 total loss，code 退化到 stage 8 才发现 | **分域 eval**：每 200 步分别算 code/funcall 的 held-out loss + 每 500 步跑 quicksort/weather smoke，binary PASS/FAIL 写到 TB |

RL 默认**跳过**（v5 v6 双双证明从 85% 起点 REINFORCE 收益 <1pp）。要跑用 `RUN_RL=1` 开短程 60 步。

---

## 1. v6 失败的真正根因

`reports/TRAINING_REPORT_8b_v6_unified.md` 的结论里把 code 退化归因于 "5.6:1 的数据不平衡"。这个数字是按 **行数** 算的（112,934 funcall vs 20,294 code），不是真实压力。

**实际上按监督 token 算，v6 的失衡是 ~30:1**：

| 维度 | funcall（glaive） | code |
|---|---|---|
| 平均行长（监督 token 数）| ~7,500 | ~1,500 |
| 行数 | 112,934 | 20,294 |
| **总监督 token** | **~847M** | **~30M** |
| **比例** | **~28:1** | — |

CE loss 是 token 级的累积量，所以模型参数被 funcall 拉走的力量，是 code 的近 30 倍。code 能力不是"逐渐遗忘"，是被 **数量级压制** 直接覆写了。

但这只是**第一个**问题。还有两个独立问题：

### 1.1 缺判别信号

v6 的 code SFT 数据（`data/sft_code/train.jsonl`）每一条都是 `<|user|> ... <|assistant|>`，**完全没有** `<|system|>` 块。funcall SFT 数据每一条都以 `<|system|>` 开头（包含 tool schema）。模型见到的训练分布是：

```
<|system|> ─────────────────────────► funcall（必须 emit <functioncall>）
（无 system）─────────────────────────► code（直接写代码）
```

这是个**完美但错误的判别函数**。模型学到的不是"用户意图"，是"system tag 的存在性"。所以推理时只要 prompt 里挂了任何 system block，模型立刻 emit `<functioncall>`，不管用户实际问什么。v5 是这个 bug 的极端版（funcall ckpt 86% rollouts 都是 functioncall），v6 没有改善这个。

### 1.2 缺监控

v6 唯一的训练时指标是 `sft/loss`（total loss 全部样本平均）。当 funcall 学得越来越好、code 越来越差时，加权平均仍然下降。v6 报告 §4.2 自己也指出这点："如果有 code-only 验证集，可能在 step 3000-4000 就能看到 code perplexity 开始上升"。但 v6 没做。

Stage 8（dual smoke）只在 RL 之后才跑，到那时已经过了 14h SFT + 4h RL = 18.5h。

---

## 2. v7 的三大设计

### 2.1 数据层：统一 conversations 格式 + token 平衡

**新文件**: `scripts/prepare_sft_v7.py`

放弃 v6 的"两个 jsonl 各自预 tokenize 再 cat 起来"做法，改成 minimind 风格的统一 conversations 格式：

```jsonl
{"source": "code",     "conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"source": "funcall",  "conversations": [{"role": "system", "content": "<...tool schemas...>"}, {"role": "user", ...}, {"role": "assistant", "content": "<functioncall> {...}"}, ...]}
{"source": "negative", "conversations": [{"role": "system", "content": "<...tool schemas...>"}, {"role": "user", "content": "<code question>"}, {"role": "assistant", "content": "<code>"}]}
```

为什么不预 tokenize：dataloader 要做 20% 概率注入，必须在 message 树上操作；并且 jsonl 直接 human-readable，调试容易。

**数据源扩充**:

| 类别 | v6 来源 | v7 来源 | v6 行数 | v7 目标 |
|---|---|---|---|---|
| code | MBPP 非 train + Codeforces 20k cap + the-stack-smol 20k cap | 同上 + **HumanEval 164** + **CodeAlpaca 20k**，cap 提到 60k | ~20k | ~140k |
| funcall | glaive-v2 全量 | glaive-v2，**cap 30k** | ~113k | ~30k |
| negative | 无 | 真 funcall system block + 真 code Q/A，**code 行数的 10%** | 0 | ~14k |

**Token 平衡的目标比例**: code+negative : funcall ≈ 1 : 1.2（脚本结尾会打印估算值，超出 0.6-1.6 区间会 WARN）。token 数用 `len(content) / 3.5`（GPT-2 BPE 在 Python 上 ~3.5 chars/token）粗估，误差 10-20%，足以指导 cap 调整。

**输出文件**:
```
data/sft_v7/train.jsonl         # 训练集，已 shuffle，~150k 行
data/sft_v7/eval_code.jsonl     # 300 行 code-only held-out
data/sft_v7/eval_funcall.jsonl  # 300 行 funcall-only held-out
```

eval shard 在 shuffle **之前**就切出去，绝不漏到训练集。

### 2.2 数据层：判别性负例 + 20% 系统提示注入

这两个是**配对的**修复，单做任何一个都不够。

#### 20% 注入（`codechat/dataloader.py: SFTConvLoader`）

来源：minimind `dataset/lm_dataset.py:9-29` 的 `pre_processing_chat`。

每次取 batch 时，对 `source != funcall|negative` 且没有 leading system 的样本，以 20% 概率前置一个**无 tools 字段**的普通 system prompt（中英混合 8 条池子）：

```
You are a helpful Python coding assistant.
You are CodeChat, a small but useful coding assistant.
你是一个乐于助人的 Python 编程助手。
...
```

效果：训练时，`<|system|>` tag 在普通 code 对话里也以 ~20% 频率出现。模型不再能用"`<|system|>` 是否存在"做判别。

#### 10% 负例（`scripts/prepare_sft_v7.py`）

随机抽一个真 glaive funcall 样本的 system block（带 tool schema），拼上一个真 code 样本的 user 问题和 assistant 答案：

```jsonl
{"source": "negative",
 "conversations": [
   {"role": "system",    "content": "You are a helpful assistant with access to the following functions...\n{\"name\": \"get_weather\", ...}"},
   {"role": "user",      "content": "Write a Python implementation of quicksort."},
   {"role": "assistant", "content": "```python\ndef quicksort(arr): ..."}
 ]}
```

模型从这种样本学到的是："**system 里挂着 tool schema ≠ 必须调用 tool**；要看 user 实际意图。"

#### 为什么必须配对

- 只做 20% 注入 → 模型学到"无 tools 的 system + code answer"，但仍然没见过"**有** tools 的 system + code answer"。推理时挂上真 schema 还是会触发 functioncall。
- 只做 10% 负例 → 训练分布里 `<|system|>` 仍然 90%+ 时间和 funcall 共现，模型还是会把 system tag 当强信号。
- **两者一起** → 判别信号被迁移到 `tools` 字段的存在性，且模型见过"有 tools 但不该用"的明确反例。

### 2.3 训练层：分域 eval + smoke 监控

**新文件**: `scripts/chat_sft_v7.py`（基于 `chat_sft.py`，FSDP/optim/ckpt 部分一字不改）

#### 分域 held-out loss（每 200 步）

每 `--eval-every` 步：
1. 把 `eval_code.jsonl` 按 rank 分片（rank `r` 处理 index 满足 `i % world_size == r`）
2. 每条样本走一次 forward，no_grad，计算监督 CE
3. all-reduce sum + count，主节点除一下取 mean
4. funcall 同理
5. 写 TB: `sft/loss_code`、`sft/loss_funcall`

eval 时 `system_inject_ratio=0.0` —— eval 必须确定性，否则 step-to-step 抖动掩盖真实 trend。

#### 推理 smoke（每 500 步）

FSDP 的 forward 会 all-gather 参数到一致状态，所以**所有 rank 跑相同 greedy decode 会得到完全相同的输出**，不需要显式 broadcast。直接所有 rank 跑同一个 prompt 的 200-token 贪心生成，主节点解析结果。

两个 smoke prompt：

```python
# Code smoke — 不带 system block
"<|user|>\nWrite a Python implementation of quicksort.\n<|end|>\n<|assistant|>\n"
# PASS 条件: 输出含 `def quicksort` / `def partition` / `def quick_sort` (re.IGNORECASE)
#           且 *不含* <functioncall>

# Funcall smoke — 带 get_weather tool schema 的 system block
"<|system|>\nYou are a helpful assistant with access to the following functions...\n{...get_weather schema...}\n<|end|>\n<|user|>\nWeather in Tokyo?\n<|end|>\n<|assistant|>\n"
# PASS 条件: 输出含 <functioncall>，能 parse 出 name=get_weather，arguments 含 location 字段
#           （不检查 location 值；v6 的 'Berlin' 幻觉是另一类问题）
```

写 TB: `sft/smoke_code_pass`、`sft/smoke_funcall_pass`（0/1）。日志同时打印生成文本前 160 字符，方便看到具体退化形态。

#### 为什么这两层都要

- Loss 是连续信号，能看 trend 但不能直接说"模型现在会不会做这个 task"。
- Smoke 是 binary 信号，直接对应"模型能不能做"，但单点抽样、可能噪声大。
- 两层一起：如果 loss_code 上涨**且** smoke_code_pass 从 1 跌到 0，确认是真实退化；如果只 smoke 抖动一次，可能是采样偶发。

---

## 3. 流水线 (`runs/train_a800_x8_v7.sh`)

5 个 stage，沿用 v6 的 `SKIP_TO=N` / `FORCE_*=1` 习惯。

| Stage | 内容 | 时间 |
|---|---|---|
| [1] | `prepare_sft_v7.py` 拉 6 个数据源 → `data/sft_v7/{train,eval_code,eval_funcall}.jsonl` | ~30min（含 HF 下载）|
| [2] | `chat_sft_v7.py` joint SFT，6000 步 (FSDP×8) | ~10-11h |
| [3] | `chat_cli` + `funcall_cli` 双 smoke（人眼读输出） | ~1min |
| [4] | （可选 `RUN_RL=1`）抽 RL prompts | <1min |
| [5] | （可选 `RUN_RL=1`）短程 funcall RL，60 步 | ~1h |

**总墙钟**: 不含 RL ≈ 11h（v6 是 18.5h），主要省在 SFT 步数 8000 → 6000 + 跳过 RL。

### 关键超参

| 名字 | v6 | v7 | 理由 |
|---|---|---|---|
| `SFT_MAX_STEPS` | 8000 | **6000** | v6 step 3000 后 loss 不再下降，后 5000 步主要在过拟合 funcall。v7 的 token 失衡修复后，没有那么多重复 funcall 要硬记，6000 步足够 |
| `SFT_LR` | 3e-5 | 3e-5 | 不变 |
| `SFT_WARMUP` | 300 | 300 | 不变 |
| `SFT_EVAL_EVERY` | — | **200** | 6000 步 ÷ 200 = 30 个 eval 点，足以看出 trend |
| `SFT_SMOKE_EVERY` | — | **500** | 12 个 smoke 点；smoke 比 eval 略重（要 200 token 生成），频率低些 |
| `SFT_SAVE_EVERY` | 500 | 500 | 不变 |
| `SFT_SYSTEM_INJECT_RATIO` | — | **0.20** | minimind 默认值 |
| `PREP_FUNCALL_CAP` | — | **30000** | 见 §2.1 token 平衡测算 |
| `PREP_NEG_FRAC` | — | **0.10** | 经验值；可视 eval 表现调节 |

### Stage 拓扑

```
checkpoints/codechat_8b/latest.pt            (预训练，train_a800_x8.sh)
        ↓  通用 SFT
checkpoints/codechat_8b_sft/latest.pt        (通用 SFT，train_a800_x8.sh)
        ↓  v7 联合 SFT (token-balanced + 判别信号 + 分域监控)
checkpoints/codechat_8b_sft_v7/latest.pt     (train_a800_x8_v7.sh stage 2)  ← 默认终点
        ↓  (可选) RUN_RL=1 短程 funcall RL
checkpoints/codechat_8b_rl_v7/latest.pt      (train_a800_x8_v7.sh stage 5)
```

v7 **不**从 v6 的 `codechat_8b_sft_v6` ckpt 续训。v6 已经向 funcall 倾斜，从那里 fine-tune 回 code 是逆向工程，不如从中性 base 重训干净。

---

## 4. 怎么判断 v7 跑得对不对

打开 TB（`tensorboard --logdir runs/tb`），关注以下曲线：

### 4.1 必须监控

| 指标 | 期望 | 出问题表现 |
|---|---|---|
| `sft/loss` | 单调下降，从 ~2.5 降到 ~0.3 | 不下降 → 数据/lr 问题 |
| `sft/loss_code` | 前 1000 步快速下降，之后稳定下降到 plateau | **持续上涨** → code 被覆写，立刻 kill |
| `sft/loss_funcall` | 同上 | 同上（此次是反过来的失衡）|
| `sft/grad_norm` | 0.5-1.0 之间稳定 | spike → 数据出现奇异样本 |
| `sft/smoke_code_pass` | 由 0/1 抖动逐步收敛到 **1**（约 step 1500-2500）| 收敛后又跌回 0 → 报告问题 |
| `sft/smoke_funcall_pass` | 同上 | 同上 |

### 4.2 关键判定

**v7 成功**的最低标准：训练结束时 `smoke_code_pass=1` **且** `smoke_funcall_pass=1` **且** `loss_code` / `loss_funcall` 都在 plateau（不是仍在快速变化）。

**v7 退化模式 A — code 被覆写**（v6 的旧病）:
- `loss_funcall` 稳定下降，`loss_code` 中后期上涨
- `smoke_code_pass` 早期偶尔为 1，后期固定 0
- 修复：降 `PREP_FUNCALL_CAP`（如 20000）或升 `PREP_NEG_FRAC`（如 0.20）

**v7 退化模式 B — funcall 被覆写**（新对称风险）:
- 反过来，code 学得太狠
- `loss_code` 下降，`loss_funcall` 上涨
- `smoke_funcall_pass` 后期为 0
- 修复：升 `PREP_FUNCALL_CAP`

**v7 退化模式 C — 判别失效**:
- 两个 loss 都正常下降，但 `smoke_funcall_pass` 在 0/1 之间反复横跳
- 含义：模型不能稳定区分"该不该 emit functioncall"
- 修复：升 `SFT_SYSTEM_INJECT_RATIO` (0.20 → 0.30) 或升 `PREP_NEG_FRAC`

---

## 5. 命令速查

```bash
# 完整跑（默认不跑 RL）
bash runs/train_a800_x8_v7.sh

# 完整跑 + 60 步 RL
RUN_RL=1 bash runs/train_a800_x8_v7.sh

# 已经准备好数据，从 SFT 开始
SKIP_TO=2 bash runs/train_a800_x8_v7.sh

# 已经训完 SFT，只跑 smoke
SKIP_TO=3 bash runs/train_a800_x8_v7.sh

# 重新生成数据
FORCE_PREP=1 bash runs/train_a800_x8_v7.sh

# 调小 funcall cap 偏向 code（如果 v7 仍然出现 code 退化）
PREP_FUNCALL_CAP=20000 PREP_NEG_FRAC=0.20 FORCE_PREP=1 bash runs/train_a800_x8_v7.sh

# 推理验证
.venv_train/bin/python -m scripts.chat_cli \
    --ckpt checkpoints/codechat_8b_sft_v7/latest.pt \
    --user "write quicksort"

.venv_train/bin/python -m scripts.funcall_cli \
    --ckpt checkpoints/codechat_8b_sft_v7/latest.pt \
    --user "Weather in Tokyo?"

# 评估
.venv_train/bin/python -m scripts.eval_funcall \
    --ckpt checkpoints/codechat_8b_sft_v7/latest.pt --num-samples 16
.venv_train/bin/python -m scripts.eval_mbpp_pass_at_k \
    --ckpt checkpoints/codechat_8b_sft_v7/latest.pt --k 8
```

---

## 6. 与 minimind 的对照

minimind 是 v7 的关键参考来源（详见与该项目交流的训练设计讨论）。对照表：

| 设计点 | minimind | v7 |
|---|---|---|
| 数据格式 | 单一 jsonl，conversations 字段 | 同 |
| Tool call 与普通对话 | 混在主线 SFT 数据 (`sft_t2t.jsonl`) | 同（`data/sft_v7/train.jsonl`）|
| Tool call 数据源 | qwen3-4b 合成 ~10w 条 + ~10 个模拟工具 | glaive-v2 真实数据，cap 30k |
| 系统提示注入 | 非 tool 样本 20% 概率加 system | 同（`SFTConvLoader.system_inject_ratio`）|
| 判别信号 | system 里有无 `tools` 字段 | 同（`source` 字段 + 内容） |
| chat template | HF tokenizer.apply_chat_template(tools=...) | CodeChat 手工 `<\|system\|>` / `<\|function_response\|>` / `<functioncall>` 拼接（沿用 v2-v6 funcall_cli 格式不动）|
| 判别性负例 | 没有显式做 | **额外加** 10% 真 system + code Q/A |
| 分域 eval | 没有显式做 | **额外加** code/funcall held-out + smoke |
| LoRA | 用于垂域微调（医疗、身份），不用于 toolcall | 不用 |

CodeChat 比 minimind 多做了两件事：判别性负例 + 分域监控。这两件不是因为 minimind 错，而是 minimind 的合成 toolcall 工具池只有 ~10 个固定工具，分布够窄，不太需要这些防御；CodeChat 用 glaive-v2 的真实 1000+ 工具时，判别难度更高，需要更强的反例信号。

---

## 7. 已知风险 / 可能的 v8 方向

### 7.1 funcall 参数幻觉

v6 smoke "Weather in Tokyo?" 输出 `{"location": "Berlin"}` —— 格式对，参数错。v7 的 reward 设计和数据都没有正面解决这个，原因：

- glaive-v2 的某些常见 city（Berlin / Paris / New York）出现频率高，模型把这些参数值"记死了"
- v7 smoke 检查 `location` 字段存在性，**不**检查值正确性，所以 v7 也会报 PASS

如果 v7 训完该问题仍存在，可以考虑：
- v8: 在 funcall reward 里加入"参数值与 user query 的语义匹配"reward（需要外部模型或字符串匹配启发式）
- 或：在 SFT 阶段对参数值做随机替换增强（每次见到 `location: "Tokyo"` 的样本就 50% 概率换成另一个 city）

### 7.2 code 数据多样性仍不足

v7 的 ~140k code 样本里，60k 来自 the-stack-smol 的 module-level docstring，质量参差。如果 MBPP pass@k 评估发现 code 质量没明显改善，下一步：
- 加 BigCodeBench、APPS-introductory 等更结构化的题库
- 加专门的 docstring → impl 数据（如 PythonCodeAlpaca、Python-Alpaca-Reasoning）

### 7.3 `eval_loss` 内部访问 SFTConvLoader 私有方法

`chat_sft_v7.py: eval_loss()` 直接调用 `loader._tokenize` / `loader._pack`。这是有意的——eval loop 需要确定性 tokenize 而非 `next_batch` 的随机抽样。如果未来 SFTConvLoader 重构，这两个方法签名要保持。

---

## 8. 文件清单

新增 / 修改的文件：

| 文件 | 类型 | 说明 |
|---|---|---|
| `codechat/dataloader.py` | **修改** | 新增 `SFTConvLoader` 类（runtime tokenization + 20% system 注入 + `deterministic_iter`），原 `SFTLoader` 不变 |
| `scripts/prepare_sft_v7.py` | **新增** | 6 个数据源整合 + token 平衡 + 负例合成，输出 conversations 格式 jsonl |
| `scripts/chat_sft_v7.py` | **新增** | 基于 `chat_sft.py`，加分域 eval + greedy smoke gen |
| `runs/train_a800_x8_v7.sh` | **新增** | 5-stage 流水线（默认不跑 RL，opt-in `RUN_RL=1`）|
| `docs/codechat_8b_v7_design.md` | **新增** | 本文档 |

**未修改**的相关文件（按 CLAUDE.md "v* 脚本 append-only" 约定）：
- `codechat/optim.py` / `codechat/checkpoint.py` / `codechat/gpt.py` / `codechat/funcall_reward.py`
- `scripts/chat_sft.py` / `scripts/chat_rl_funcall.py` / `scripts/funcall_cli.py` / `scripts/chat_cli.py`
- `scripts/prepare_sft.py` / `scripts/prepare_sft_code.py` / `scripts/prepare_sft_funcall.py`
- 所有 `runs/train_a800_x8_v[1-6]*.sh`
