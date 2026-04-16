# 混合 SFT vs MoE：同一思路吗？

v6 用 `sft_code + sft_funcall` 混合数据联合训练一个模型，同时获得写代码和调工具两种能力。这和 **MoE (Mixture of Experts)** 是一回事吗？

**一句话答案**：**不是**。动机有重叠（让一个模型拥有多种能力），但实现层完全不同 —— 一个是**数据工程**，一个是**架构工程**。

## 1. 两者到底在做什么

### 混合 SFT（v6 的做法）

```
  ┌────────────────────────────────────┐
  │       单一 8B 模型，稠密           │
  │                                    │
  │   所有参数对每个 token 都参与      │
  └────────────────────────────────────┘
           ▲              ▲
           │              │
      code 数据        funcall 数据
       (shuffled together)
```

- 架构：**不变**，还是普通 transformer
- 参数：**全部共享**，每个 token 都走完整的前向
- 任务路由：**靠 prompt 格式**（v6 里是 `<|system|>` 工具块的有无）
- 训练：**只改数据**，混合两种分布的 SFT 样本后 shuffle

### MoE

```
  ┌─────────────────────────────────────────────┐
  │  Router 决定每个 token 走哪 k 个 expert    │
  │                                             │
  │   Expert 1   Expert 2  ...  Expert N       │
  │   [FFN]      [FFN]          [FFN]          │
  │                                             │
  │   共享注意力 / embedding                    │
  └─────────────────────────────────────────────┘
           ▲
           │
      任意混合数据
```

- 架构：**改了**，FFN 层换成 N 个并行 expert + 一个可训练 router
- 参数：**总量很大**（N 倍 FFN），但每个 token 只激活 k 个（通常 k=2, N=8~128）
- 任务路由：**router 学出来**的，按 token 粒度动态路由
- 训练：loss 里常加 router 均衡正则（load balancing loss），否则 router 退化成只用一两个 expert

## 2. 关键差异对照

| 维度 | 混合 SFT | MoE |
|---|---|---|
| **激活模式** | 所有参数稠密激活 | 稀疏，每 token 只用 k/N 参数 |
| **总参数量** | 保持不变（8B 就是 8B） | 大很多（8×8B FFN ≈ 几十 B） |
| **推理 FLOPs** | 和单模型一样 | 约等于 (k/N) × 稠密等价模型 |
| **显存** | 单模型显存 | 总参数全部要存，FLOPs 省但显存省不多 |
| **任务路由** | **显式**：由 prompt 格式决定（人工 protocol） | **隐式**：router 网络学出来 |
| **路由粒度** | prompt 级（整条对话一个 mode） | token 级（每个 token 可能去不同 expert） |
| **训练代价** | 就是多了点数据 | 需要重新设计架构 + 从头训或做 upcycling |
| **专家边界** | 没有物理边界，两种能力在参数里**纠缠** | 物理边界清晰（router 可以看） |
| **容灾性** | 一种任务训坏会连累另一种 | Expert 间相对隔离，单个 expert 崩不影响其他 |

## 3. 共同动机

两者都在回答同一个问题：**一个模型能否同时掌握多种能力？**

更深一层，都在对抗**容量稀释**：

> 如果一个 FFN 要同时装下「写 Python」「调函数」「懂法语」「做数学」，每种能力分到的参数都被稀释了。容量够不够用，决定了多任务到底能不能兼得。

- **混合 SFT 的赌注**：8B 的容量足够装下几种相关能力，靠数据分布教会模型"看 prompt 切换 mode"。
- **MoE 的赌注**：不够就直接堆容量，但只在需要时激活对应的 expert，FLOPs 不涨。

## 4. 类比：公司组织结构

| 方案 | 类比 |
|---|---|
| 单任务 SFT | 雇一个只会写代码的程序员 |
| 混合 SFT | 雇一个全栈工程师（同一个人做两种活，能力之间会互相影响） |
| MoE | 雇一个项目经理 + N 个专家（PM 根据问题把活派给最合适的专家） |

全栈工程师可能没有纯专家精，但沟通成本低、招聘成本低。MoE 的专家更精，但需要一个好 PM（router），否则 PM 老是派活给同一个人（load imbalance）。

## 5. 什么时候用哪个

**用混合 SFT：**
- 任务之间有**语义重叠**或**共享底层能力**（写代码和调工具都需要结构化输出能力）
- 推理预算固定，不想改架构
- 训练数据每种任务都有几万到几十万条
- 只有几种任务（2-5 种），不会把容量稀释到不够

**用 MoE：**
- 任务差异非常大（代码 + 图像描述 + 翻译 + 数学）
- 想在固定推理 FLOPs 下**扩大模型知识容量**
- 有预算做架构改造和更长的训练
- 任务数量很多（10+ 种 domain）

## 6. v6 为什么选混合 SFT 而不是 MoE

1. **能力差异小**：写代码 vs 调工具，两者都是**结构化文本输出**，底层能力（token prediction、格式保真）是共享的。不需要强制隔离。
2. **prompt 边界清晰**：有 `<|system|>` 工具块就走 funcall，否则写代码 —— 路由信号太明确，学一个 router 属于杀鸡用牛刀。
3. **架构改造成本高**：MoE 需要改模型代码、重新做 FSDP sharding、设计 load balance loss、处理 expert-parallelism，研发周期几周起。混合 SFT 改一行 `cat` 命令。
4. **容量够用**：8B 稠密容量装下两种相关能力大概率够（glaive 113k + code 30k ≈ 143k 样本，参数冗余很大）。
5. **v5 证明了路径可行**：v5 纯 funcall SFT 能跑到 86% 全匹配，说明这套 tokenizer + 数据管线对 funcall 格式学得很扎实。v6 只是在此基础上**再塞点代码数据**，风险可控。

## 7. 能不能两个一起用？

能。现实里的做法是：

```
MoE 架构 + 混合数据 SFT
```

DeepSeek-V3、Mixtral-8x7B-Instruct、Qwen2-MoE 等都是**MoE 架构上做混合 SFT**。
- MoE 提供稀疏容量
- 混合数据提供多任务监督
- Router 学到"这类 token 交给 expert 3，那类 token 交给 expert 5"

对 CodeChat 来说这是未来的方向，但现在 8B 稠密模型还有很多没榨干的优化空间（数据配比、RL 奖励设计、在线评测），先把混合 SFT 的路走清楚再说。

## 8. 容易搞混的地方

误解：**"混合 SFT 是 MoE 的简化版，router 变成人肉 prompt 就是 MoE。"**

不对。MoE 的本质是 **sparse activation** —— 推理时只激活一部分参数以省 FLOPs。混合 SFT 没有任何稀疏性，**所有参数每个 token 都在跑**。两者解决的问题不同：

- **MoE 解决**：想要更大容量但不想付对应 FLOPs 代价
- **混合 SFT 解决**：想让一个定容量模型覆盖多个任务分布

不同层的事情，不是同一思路的两个实现。

## 9. 参考

- MoE 经典：Shazeer et al. 2017《Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer》
- Mixtral-8x7B：Jiang et al. 2024《Mixtral of Experts》
- 混合 SFT 的理论基础：Wei et al. 2022《Finetuned Language Models Are Zero-Shot Learners》(FLAN) —— 证明了多任务混合 SFT 的泛化收益
- MathGPT 的 distribution-matching SFT + RL：本仓库 `runs/train_a800_x8_v5_funcall.sh` 第 1-60 行头注释
