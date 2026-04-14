# CodeChat vs MathGPT 对比报告

> **作者视角**：两个项目都是基于 nanochat 风格的"预训练 → SFT → RL"三段式训练框架，但最终 RL 阶段一个失败（reward ≡ 0.000，415 步）、一个成功（reward 0.07 → 0.41+，Pass@1 14.75%）。本报告对比两者的**代码实现**、**训练配置**和**任务/奖励设计**，定位失败的根因。
>
> 参照报告:
> - CodeChat: `reports/TRAINING_REPORT_8b_a88_x8.md`
> - MathGPT:  `../MathGPT/reports/A800_TRAINING_REPORT_v2.md`

---

## 一、项目定位对比

| 维度 | CodeChat | MathGPT |
|---|---|---|
| 任务域 | Python 代码生成（MBPP 单元测试） | 小学数学题（GSM8K 数字答案） |
| 基础仓库 | 自写 GPT（`codechat/gpt.py`）+ 自写 tokenizer | fork 自 nanochat，复用其 Engine / tokenizer / checkpoint |
| 模型规模 | **8B**（depth=40, n_embd=4096），bf16 | **~560M**（d20, 20 层），bf16 |
| 并行方式 | FSDP FULL_SHARD × 8 A800 | DDP × 8 A800（nanochat 原生） |
| 优化器 | AdamW（全部参数） | MuonAdamW（矩阵用 Muon，emb/head 用 Adam） |
| 实际 RL 结果 | **reward ≡ 0.000**（415 步无信号） | **reward 0.41 起步，Pass@1 峰值 14.75%** |

---

## 二、代码实现差异

### 2.1 RL 主脚本结构对比

| 项 | CodeChat (`scripts/chat_rl.py`, 370 行) | MathGPT (`scripts/train_rl.py`, 312 行) |
|---|---|---|
| 算法 | GRPO：`adv = (r − r̄) / (σ + ε)` | REINFORCE with baseline：`adv = r − r̄`（无标准化）|
| 参考模型 | **有** frozen ref model，KL penalty（`kl_coef=0.02`）| **没有** ref 模型，**没有 KL 项** |
| Rollout 组织 | 单 prompt × `group_size=4` 次 sample（串行 for 循环） | 单 prompt × `num_samples=32`（Engine.generate_batch 一次出来） |
| 数据集 | MBPP sanitized train，~374 条 | GSM8K train，~7,500 条 |
| 奖励函数 | 子进程执行单元测试（`run_with_tests`） | 正则提取 `#### N` 后字符串相等（`extract_answer`） |
| 奖励粒度 | 可选 binary / fractional / tiered（三档阶梯） | 纯 binary (1.0 or 0.0) |
| Eval | 没有在线 pass@k eval（RL loop 中） | **每 30 步跑 400 题 pass@k**（早停依据） |
| 每步耗时 | ~145 s（4 次 384-token generate + 4 次 forward） | ~80 s（32 次生成打包成一个 batch） |

### 2.2 采样实现差异（关键）

**CodeChat** — 一次一条 token 的 Python 循环：

```python
# scripts/chat_rl.py:100-133
def sample_one(model, prompt_ids, max_new, temperature, top_k, ...):
    for _ in range(max_new):
        cond = ids[:, -block_size:]
        logits, _ = model(cond)
        ...
        nxt = torch.multinomial(probs, 1)
        if is_dist:
            dist.broadcast(nxt, src=0)   # 所有 rank 同一 token
        ...

# 主循环里再串行 4 次：
for _ in range(args.group_size):
    new_ids, _, full_ids = sample_one(...)
```

每个 token 一次 FSDP 前向 + `dist.broadcast`，**8 卡全在做重复计算**（结果强制同步）——FSDP 在这里是纯负担。

**MathGPT** — 整批次并发生成：

```python
# scripts/train_rl.py:128-136
for sampling_step in range(args.num_samples // args.device_batch_size):
    seqs_batch, masks_batch = engine.generate_batch(
        tokens,
        num_samples=args.device_batch_size,
        max_tokens=args.max_new_tokens, ...
    )
```

调用 nanochat 的 `Engine.generate_batch`——一次前向出一批，不同 rank 处理不同 `ddp_rank` 的 example，天然并行。

### 2.3 奖励函数差异（**决定成败的代码**）

**CodeChat `codechat/execution.py`**（MBPP 单元测试）：

```python
def run_with_tests(code, tests, mode="fractional") -> float:
    # 1) extract_code 从 ```python ... ``` 里扒代码
    # 2) ast.parse 失败 -> 0.0
    # 3) 起子进程 exec(code) 再跑每个 assert
    # 4) 三种 mode：
    #    - binary:     全过 1.0，否则 0.0
    #    - fractional: passed / total
    #    - tiered:     阶梯式（0.05 可解析 / 0.15 可 exec / 0.15+0.85*k/n 部分通过 / 1.0 全过）
```

**注意**：报告里 RL 实际跑的是 `binary` 模式（最严格），所以 4 个 rollout 全 0 是常态。

**MathGPT `tasks/gsm8k.py`**（数字答案匹配）：

```python
GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(completion):
    match = GSM_RE.search(completion)
    return match.group(1).strip().replace(",", "") if match else None

def reward(self, conversation, assistant_response):
    ref_num  = extract_answer(ground_truth_answer)
    pred_num = extract_answer(assistant_response)
    return float(pred_num == ref_num)    # 字符串相等即给 1.0
```

**两者难度的天壤之别**：
- CodeChat：模型要生成**完整可运行的 Python 函数** + 通过**多个 assert**，任何一个语法错、缩进错、import 错都归零。
- MathGPT：模型只要在回答末尾写出 `#### 42`，数字对就给 1.0。**格式门槛极低**，一个乱猜的数字有时都能撞中。

### 2.4 SFT 数据对齐 RL 任务的程度

**CodeChat `scripts/prepare_sft.py`**：

```python
# 两个来源，全是"通用 Python 指令→回答"的 Alpaca 格式：
load_dataset("iamtarun/python_code_instructions_18k_alpaca")  # 18k
load_dataset("sahil2801/CodeAlpaca-20k")                       # 20k
# 总 38k，**完全没有 MBPP 风格的数据**（MBPP = 自然语言描述 + 隐藏的 assert test）
```

**MathGPT `scripts/chat_sft.py:174-182`**：

```python
train_tasks = [
    SmolTalk(...),                                    # 460K 通用对话
    CustomJSON(identity_conversations_filepath),      # 1k×2 身份
    *[MMLU(...) for _ in range(mmlu_epochs)],        # 100K × 5 = 500K
    *[GSM8K(subset="main", split="train") for _ in range(gsm8k_epochs)],  # 8K × 16 = 128K
    SimpleSpelling(size=200000, ...),
    SpellingBee(size=80000, ...),
]
```

**关键**：MathGPT SFT **直接把 GSM8K train 拿来训 16 个 epoch**（每条看 ~96 次），让模型学会了"写推理 → 末尾 `#### N`"的**精确输出格式**。而 CodeChat SFT 从未在 MBPP 的 `prompt → 可运行代码 + 通过 assert` 分布上训过。

---

## 三、训练配置差异

### 3.1 SFT 阶段

| 参数 | CodeChat 8B | MathGPT v2 |
|---|---|---|
| 基础步数 | 3,000 | 2,999 |
| Batch tokens | 131k (1×8×8×2048) | 65k (32×2048) |
| LR | 5e-5 cosine | 继承预训练 × 0.8，warmdown 到 0 |
| 数据总量 | 38k examples | 1.36M rows (含 128K GSM8K) |
| 每条数据被看的次数 | **~1×**（38k / 3000 步，1 步看多行但一个 epoch 内）| **~96×**（GSM8K 被反复刷）|
| 最终 loss | 0.116 | val BPB 0.354 |
| RL 任务数据是否在 SFT 里 | **❌ 完全没有** | **✅ GSM8K × 16 epochs** |

### 3.2 RL 阶段

| 参数 | CodeChat 8B | MathGPT v2 |
|---|---|---|
| 采样数 / prompt | **4** | **32**（8×）|
| Max new tokens | 384 | 768 |
| Temperature / top-k | 0.9 / 50 | 1.0 / 50 |
| Advantage | `(r−r̄)/(σ+ε)` GRPO | `r−r̄` REINFORCE |
| KL penalty | 0.02 × mean(logp−logp_ref) | **无** |
| LR | 1e-5（绝对值） | init_lr_frac=0.02 × base lr（相对值）|
| 在线 eval | 无 | pass@k on 400 GSM8K test examples，每 30 步 |
| 停止准则 | max_steps=1000（死跑） | 按 pass@k 峰值决定 best ckpt |
| 初始 reward | **0.000** | **0.41** |
| 实际结果 | reward 停在 0，415 步无信号 | Pass@1 从 5.5% → 14.75% (step 480) |

### 3.3 分布式实现差异

**CodeChat**（必须 FSDP，否则 OOM）：
- 8B 参数 fp32 AdamW 状态 ~96GB，单张 80GB 卡放不下
- FSDP FULL_SHARD 把 params/grads/optim 切 8 份
- **副作用**：RL 的 `sample_one` 每 token 都得 `dist.broadcast` 同步，8 卡全做重复计算

**MathGPT**（DDP 即可）：
- 560M 模型 + Muon 优化器，单卡完全够
- 每个 rank 处理独立 GSM8K 子集（`range(ddp_rank, len(task), world_size)`）
- **梯度天然是 8 条不同数据的平均**，信号多样性 × 8

---

## 四、为什么 CodeChat RL 失败 & MathGPT RL 成功

### 4.1 失败的数学机制

GRPO 的梯度：

```
∇θ J = advantage × ∇θ log π(a|s)
advantage = (r − r̄) / (σ + ε)
```

当 group 内 4 个 rollout 的 reward **全是 0**：
```
r̄ = 0, σ = 0 → advantage = 0 / (0 + 1e-6) ≈ 0 → 梯度 ≡ 0
```

此时训练 loss ≈ 0.03 是**纯 KL penalty 的自振荡**（`kl_coef × mean(logp_policy − logp_ref)`），不是在学习。415 步下来模型只是在 KL 项的微弱拉扯下漂移。

### 4.2 三层根因（从直接到根本）

#### 根因 1：Base 能力 × 任务门槛不匹配（最关键）

| 条件 | CodeChat | MathGPT |
|---|---|---|
| 任务 pass rate（SFT 后估计）| **<1%**（3.93B tokens 预训 + 通用 Alpaca SFT 写不出过 MBPP 的 Python） | ~5-10%（SFT 在 GSM8K 上训过 16 个 epoch）|
| group 全 0 概率（pass rate 1%, G=4）| **~96%** | — |
| group 全 0 概率（pass rate 5%, G=32）| — | **~19%**（81% 的 step 有非零信号）|

**一句话**：CodeChat 的 base 根本没到 MBPP 这个任务的"起跳线"，GRPO 没有任何梯度信号可学。

#### 根因 2：Reward 设计（第二层放大器）

| 设计 | CodeChat（binary 实跑） | MathGPT |
|---|---|---|
| 奖励类型 | 全部 unit test 通过才 1，否则 0（**阶梯 reward 代码写了但没用**）| 数字字符串匹配，1 或 0 |
| 真实信号源 | 需要"完整可运行 Python + 通过 N 个 assert" | 需要"末尾写出对的数字" |
| 容错 | 零（语法错、import 错、函数名写错全归 0）| 模型可以写错中间推理过程，只要最后数字对就算对 |

CodeChat 的 `execution.py` **已经实现了 tiered reward**（0.05 可解析 / 0.15 可 exec / 部分通过 / 1.0 全过），但报告里 RL 实跑用的是 `binary`——把最便宜的破 0 工具关掉了。

#### 根因 3：SFT 数据 × RL 任务**完全错配**（最深的结构性问题）

- CodeChat SFT = 通用 Python Alpaca 指令回答（`instruction → output`）
- CodeChat RL  = MBPP（`自然语言描述 → 能通过隐藏 assert 的函数`）
- **两者的输出分布根本不是一回事**。Alpaca 是"讲一段话+贴一段代码"，MBPP 要求"精确的函数签名 + 可执行"。

对比 MathGPT：SFT 里 **16 个 epoch 刷 GSM8K**，RL 也是 GSM8K。SFT 阶段模型已经学会"写推理链 → `#### 42`"，RL 只需要把已有能力微调到更高 pass rate。

### 4.3 MathGPT 成功的三个额外加成

1. **num_samples=32**（CodeChat 只有 4）：pass rate 5% 时，group=32 至少一个对的概率是 81%，给了稳定的梯度方差。
2. **`examples_per_step=32` × DDP**：8 个 rank 各跑自己的 example，每个 step 梯度来自 32 × 8 = 256 条 rollout，信号密度 >>> CodeChat 的 4 条。
3. **在线 pass@k eval + checkpoint 轮转**：每 30 步 eval 400 题，知道最优点在 step 480 而不是盲跑到末尾；CodeChat RL 没有在线 eval，只能训完再 eval。

---

## 五、修复路线（对 CodeChat RL）

按性价比递减排序（对应 `TRAINING_REPORT_8b_a88_x8.md §5.6`）：

### 一等：立刻可做，成本近零

1. **把 `--reward-mode` 从 `binary` 改到 `tiered`** — 代码已经写好，只是训练脚本传参没用。tiered 模式下语法正确就给 0.05，能 exec 给 0.15，直接打破 group 全 0。
2. **`group_size` 从 4 提到 8-16** — pass rate 2% 时 group=4 全 0 是 92%，group=16 降到 73%。
3. **打开 `--log-rollouts-every 50`** — 肉眼看 rollout 在写什么。代码里已经有了。
4. **先跑 `scripts/eval_mbpp_pass_at_k.py`** — 半小时确定 SFT ckpt 的真实 pass@k，<1% 就别开 RL（代码已存在）。

### 二等：要重跑 SFT

5. **SFT 数据加 MBPP train 的标准解**（参考 MathGPT 把 GSM8K 直接刷进 SFT 的做法）。`scripts/prepare_sft_code.py` 已经存在，把它加进 pipeline。
6. **用 `scripts/filter_mbpp_by_passrate.py` 过滤数据集** — 只留 pass rate ∈ [0.1, 0.9] 的题，每条都有非零信号。

### 三等：策略转向

7. **暂时放弃 MBPP，跑 funcall SFT**（`runs/train_a800_x8_v2_funcall.sh`）。funcall 是格式任务、监督稠密、不依赖生成质量——和 MathGPT 成功的原因（任务简单 + SFT 分布与下游对齐）是同一套逻辑。

---

## 六、核心启示

1. **RL 不是魔法**：它只能把"模型偶尔能做对"放大成"经常做对"。如果 SFT 后任务 pass rate ≈ 0，RL 没有任何可学的东西。
2. **SFT 数据必须覆盖 RL 的任务分布**：MathGPT 把 GSM8K train 塞进 SFT 16 轮是灵魂操作。CodeChat 用通用 Alpaca SFT + MBPP RL 是分布错配。
3. **Binary reward 是 GRPO/REINFORCE 的"死亡陷阱"**：任何稀疏二元奖励都应该配对 group_size ≥ 16 和 dense / shaped reward 保底。CodeChat 已经实现了 tiered reward 却没有启用，是最可惜的一点。
4. **小模型 + 简单任务 + 对齐 SFT + 密集 eval** > **大模型 + 难任务 + 错配 SFT + 盲跑**。MathGPT 的 560M 在 GSM8K 上拿到 Pass@1 14.75%，CodeChat 的 8B 在 MBPP 上 reward≡0——模型规模不能弥补设计缺陷。
5. **在线 eval 是 RL 的刹车**：没有它就是盲跑。MathGPT 每 30 步 400 题 pass@k 的代价很低，产出了"best ckpt 在 step 480"这样的可行动结论。CodeChat 应当补上 `--eval-every` 逻辑。

---

## 七、一句话总结

> **CodeChat 的 8B 在 MBPP 上 reward≡0，不是 GRPO 坏了，也不是 FSDP 坏了，是 `base能力 × 奖励门槛 × SFT分布` 这三者的乘积等于零。MathGPT 的成功来自 `适当难度 × 密集奖励 × 对齐SFT` 三个条件全部满足——它不是更强的模型，而是更正确的训练设计。**
