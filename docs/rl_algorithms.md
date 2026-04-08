# CodeChat RL 算法选型

本文档回答两个问题：
1. CodeChat 当前为什么用 GRPO？
2. 在「Python 代码问答 + 可执行奖励」这个具体场景下，是否有比 GRPO 更合适的 RL 算法？

## 1. 场景特点先说清楚

CodeChat 的 RL 场景有几个非常特殊的属性，这些属性直接决定了算法选型：

| 特点 | 含义 | 对算法的影响 |
|---|---|---|
| **奖励可验证** | `assert` 跑通就是 1，跑不通就是 0，无噪声无主观 | 不需要 reward model，避免 reward hacking |
| **奖励稀疏且二值化** | 每题只在生成结束后拿到一个 scalar，中间无过程信号 | 价值函数 (critic) 很难学 |
| **奖励方差大** | 同一题不同采样之间经常 0/1 跳变 | 需要方差缩减技巧（baseline、组归一化） |
| **长序列生成** | 一次 rollout 几百 token | PPO 的 per-token critic 很昂贵且不稳 |
| **单卡训练** | A800 × 1，显存紧 | 不能同时常驻 policy + critic + ref 三个 2B 模型 |
| **采样成本高** | 2B 模型采一条 384 token 要几秒 | 算法必须 sample-efficient |

## 2. 为什么先选 GRPO

**GRPO (Group Relative Policy Optimization)** 是 DeepSeek 在 DeepSeekMath / DeepSeek-R1 里用的算法，核心思想是：

> 对同一个 prompt 采 G 条回答，把组内奖励归一化 `(r - mean)/std` 当作 advantage，直接丢进 PPO-clip 目标，**完全不要 critic**。

它刚好命中上面每一条痛点：

1. **不要 critic** → 显存里只需要 policy + frozen reference（SFT ckpt），2B 模型在 80GB 上才装得下。PPO 要多一个同等大小的 value head，直接爆。
2. **组内 baseline** → 对二值/稀疏奖励天然是很强的方差缩减，比学一个独立 critic 稳。对代码 0/1 奖励尤其合适：只要组内有一条对、一条错，就有非零学习信号。
3. **可验证奖励** → 不需要 RLHF 的 reward model，避免 reward hacking，也省掉了 reward model 的训练和存储。
4. **实现极简** → `codechat/scripts/chat_rl.py` 核心就一百来行；没有 critic loss、没有 GAE、没有 value clip。
5. **与 SFT 同源** → KL 到 ref 模型用 k1 近似 (`logπ - logπ_ref`)，防止策略跑偏忘掉 SFT 学到的代码格式。

一句话：**GRPO = 「PPO 去掉 critic，换成组内归一化」**，这对代码这种「可验证 + 稀疏 + 单卡」场景是当前的甜点。

## 3. 候选算法横评

### 3.1 PPO（经典 RLHF）

**做法**：actor + critic 双模型，critic 学 value，advantage 由 GAE 算出来，actor 用 clip surrogate。

**对 CodeChat 的问题**：
- Critic 是一个和 policy 同规模的头，2B policy + 2B critic + 2B ref ≈ 12GB×3 权重 + 3×AdamW 状态，单卡 A800 装不下
- 二值稀疏奖励下 value 学得很慢、很抖，GAE 估出来的 advantage 噪声比 GRPO 的组归一化还大
- 实现复杂：value loss、GAE、两份优化器

**结论**：在大模型 + 可验证奖励场景已经被 GRPO/RLOO 全面替代。PPO 在连续控制、稠密奖励的场景依然是金标准，但不是这里。

### 3.2 RLOO（REINFORCE Leave-One-Out）

**做法**：同样每个 prompt 采 G 条，但 baseline 用「除自己外其他 G-1 条的平均奖励」，不除以 std。目标函数是纯 REINFORCE（没有 clip）。

**对比 GRPO**：
- **优点**：
  - baseline 理论上无偏（GRPO 的 std 归一化是有偏的）
  - 没有 ratio clip，不会在高 ratio 时丢梯度
  - 近期多篇论文（Ahmadian et al. 2024 "Back to Basics"）实验指出在 RLHF 上 RLOO ≥ PPO ≥ GRPO
- **缺点**：
  - 没有 clip → 需要更小的 lr，或者额外的 KL 约束防止更新过激
  - 对离 policy 严重的样本没有保护（GRPO 的 clip 在这里有用）

**结论**：**RLOO 在代码场景可能比 GRPO 更稳**，尤其是组内奖励方差不大时。值得作为第二个实现选项。

### 3.3 DPO / KTO（离线偏好优化）

**做法**：不采样，用静态的 `(chosen, rejected)` 对直接做对比损失。

**对比 GRPO**：
- **优点**：无采样、无执行器、训练速度快 10×
- **致命缺点**：**拿不到可执行信号**。代码正确与否只有跑一次才知道，DPO 需要预先构造偏好对；你要么用一个更强的老师模型来标（回到 RLHF 老路），要么用固定参考答案（只教"这样写"，学不到"为什么这样写能过测试"）
- 经验上 DPO 在代码任务上的提升远小于在一般对话上的提升

**结论**：DPO 适合"对齐风格 / 偏好"，不适合"最大化通过率"。可以作为 SFT 和 GRPO 之间的一个低成本中间步骤，但不能取代 GRPO。

### 3.4 Expert Iteration / STaR / Rejection-sampling Fine-tuning (RFT)

**做法**：采样 → 只保留跑通测试的回答 → 当作新的 SFT 数据再训一轮 → 循环。

**对比 GRPO**：
- **优点**：
  - 实现最简单，就是 SFT，不需要 PG loss、KL、ref model
  - 训练非常稳，没有策略崩溃风险
  - 显存和 SFT 一样
- **缺点**：
  - **只利用正样本，扔掉所有失败 rollout 的信号**。GRPO 能从失败里学（推失败概率），RFT 不能
  - 在初始通过率很低时进度慢（0 条对 → 没数据）
  - 容易过拟合到"碰巧对的"短回答

**结论**：RFT 是一个**非常强的基线**，DeepMind 的 AlphaCode、Google 的 STaR、Meta 的 Self-Taught Reasoner 都在用。在 CodeChat 初期 SFT 模型通过率还不低的前提下，RFT 能用 1/5 的代码量拿到 GRPO 80% 的收益。**建议作为 RL 之前的一个前置热身阶段**。

### 3.5 PPO + Process Reward Model (PRM)

**做法**：不仅对最终 0/1 打分，还对生成过程中的每一步 / 每一行代码由 PRM 打中间分，让 RL 拿到稠密信号。

**对比 GRPO**：
- **优点**：稠密奖励 → value 学得动 → PPO 重新可用；能学到"走到一半就知道方向对不对"
- **缺点**：要先训一个 PRM，代码场景的过程奖励很难定义（第几行算"进展"？），成本高
- DeepSeek-R1 的经验：结果奖励 + GRPO 比 PRM + PPO 更简单、效果更好，PRM 容易被 reward hack

**结论**：工程复杂度和收益不划算，不选。

### 3.6 ReST / ReST-EM（Google DeepMind）

**做法**：交替做「采样扩库 (Grow)」和「在扩充后的库上 SFT (Improve)」，多轮迭代。本质是 RFT 的加强版，加入了温度退火和过滤阈值。

**结论**：和 RFT 类似，适合和 GRPO 互补而不是取代。

## 4. 针对 CodeChat 的推荐路线

综合上面的分析，在**单卡 A800 80GB + 2B 模型 + 可执行奖励**的硬约束下：

| 优先级 | 方案 | 理由 |
|---|---|---|
| **现在 (v1)** | **GRPO**（当前实现） | 显存能装下，实现极简，组内 baseline 对 0/1 奖励稳 |
| **下一步 (v1.5)** | **RFT 前置** + GRPO | 先用几百步 RFT 把通过率从 low baseline 拉起来，再 GRPO 精修，收敛更快 |
| **可选替换 (v2)** | **RLOO** | 如果观察到 GRPO 的 clip 频繁触发、或 std 归一化让学习率不好调，改 RLOO |
| **不建议** | PPO / PPO+PRM / 纯 DPO | 显存、复杂度、或与场景不匹配 |

换句话说：**GRPO 是当前的最佳默认，RLOO 是最值得尝试的替代，RFT 是最值得前置的热身，其它都是弯路。**

## 5. 如果要在 CodeChat 里换成 RLOO，需要改什么

`scripts/chat_rl.py` 只有两处要改：

1. Advantage 计算：
   ```python
   # GRPO:
   adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
   # RLOO:
   G = rewards.shape[0]
   loo_mean = (rewards.sum() - rewards) / (G - 1)
   adv = rewards - loo_mean
   ```
2. 损失：去掉 ratio clip（我们当前实现里本来就没写 clip，只是纯 PG），把 `kl_coef` 稍调大（例如 0.04）来补偿缺少 clip 的稳定性。

其它所有东西（执行器、采样、ref model、KL）都不变。这也是为什么说 GRPO ↔ RLOO 的切换成本几乎为零——**真正贵的是执行器和采样，不是 loss 形式**。

## 6. 参考

- DeepSeek (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* —— 提出 GRPO
- DeepSeek-AI (2025). *DeepSeek-R1* —— 纯 GRPO + 结果奖励训练推理模型
- Ahmadian et al. (2024). *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs* —— RLOO 论文
- Zelikman et al. (2022). *STaR: Self-Taught Reasoner* —— RFT 思想
- Singh et al. (2023). *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models* —— ReST-EM
- Rafailov et al. (2023). *Direct Preference Optimization* —— DPO
