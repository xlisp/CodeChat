# Mask、三角矩阵与因果掩码：历史、起源与原理

> 本文以 CodeChat 仓库中的实际代码（`codechat/gpt.py`、`codechat/dataloader.py`）为主线，
> 辅以 HuggingFace `transformers` 库的对应实现，讲清楚三个纠缠在一起、却经常被混为一谈的概念：
> **mask（掩码）**、**三角矩阵（triangular matrix）**、**causal mask（因果掩码）**。
>
> 一句话先摆在这里：在本仓库里，因果掩码被 PyTorch 的一行 `is_causal=True` 隐藏掉了
> （`codechat/gpt.py:67`），三角矩阵被藏进了内核里；而另一种完全不同的 mask——**loss mask**
> （`ignore_index=-100`，`codechat/gpt.py:132`、`codechat/dataloader.py`）——却显式地散落在数据管线各处。
> 把这两种 mask 分清楚，是读懂整个训练流程的前提。

---

## 0. 两种 mask，别搞混

"mask" 这个词在 Transformer 语境里至少指两件毫不相干的事，本仓库两者都用：

| | 因果掩码 (attention mask) | 损失掩码 (loss mask) |
|---|---|---|
| 作用对象 | 注意力分数矩阵 `QKᵀ`（`T×T`） | 交叉熵的 `targets`（`T`） |
| 目的 | 禁止 token 看见"未来" | 禁止某些 token 参与梯度 |
| 形状 | 二维三角矩阵 | 一维向量 |
| 本仓库位置 | `is_causal=True`（`gpt.py:67`） | `-100` / `ignore_index=-100`（`gpt.py:132`、`dataloader.py:81` 等） |
| 值语义 | `0`（可见）/ `-∞`（屏蔽） | 真实 token id（计损）/ `-100`（忽略） |

前者是**结构性**的、每个序列都一样、是"因果性"这条物理定律的体现；
后者是**内容性**的、每条样本不同、决定"这句话里哪几个字该被学"。
本文主体讲前者（因果掩码 + 三角矩阵），第 5 节回到后者，因为它才是本仓库里唯一"看得见"的 mask。

---

## 1. Mask 的起源：从信号处理到神经网络

"mask" 一词并非深度学习发明。它最早来自**摄影制版与信号处理**：一张遮片（mask）挡住一部分光/信号，只让你想要的部分通过。计算领域沿用了这个隐喻：

- **位掩码（bitmask）**：`x & 0b1010` 只保留特定的位，是计算机科学里 mask 最古老的用法，早于神经网络几十年。
- **图像处理的卷积掩码 / 核（kernel）**：一个小矩阵滑过图像，"遮住"邻域外的像素。
- **序列模型里的 padding mask**：把不等长序列补齐到同一长度后，用 mask 标记"这些位置是填充的、别算进去"。

到了神经网络时代，mask 演化出两条主线，正好对应第 0 节的两列：

1. **屏蔽计算**——在 attention 里，把某些位置的分数压到 `-∞`，softmax 之后权重变 0。这是**因果掩码**的技术手段。
2. **屏蔽损失**——在 loss 里，把某些位置标记为"不计梯度"。这是**loss mask**，PyTorch 用魔数 `-100` 实现。

因果掩码真正成名，是在 2017 年的论文 **《Attention Is All You Need》**（Vaswani et al.）。它把"未来不可见"这一自回归语言模型的根本约束，第一次干净地表达成了一个作用在注意力矩阵上的**下三角遮罩**。原文称之为 *"masked self-attention"*，并强调："我们通过在缩放点积注意力内部屏蔽（置为 −∞）所有对应非法连接的值来实现这一点。"

---

## 2. 三角矩阵：一个古老的线性代数工具被重新起用

**三角矩阵**（triangular matrix）远比深度学习古老，是 19 世纪线性代数的基本对象：

- **下三角矩阵（lower triangular）**：主对角线以上全为 0。
- **上三角矩阵（upper triangular）**：主对角线以下全为 0。

它们在数值计算里无处不在——**LU 分解**、**Cholesky 分解**、解线性方程组的前代/回代，都建立在三角矩阵"一个变量只依赖前面已解出的变量"这一性质上。注意这句话：**"只依赖前面的"**——这正是自回归语言模型需要的性质。因果掩码的天才之处，就是发现了"token t 只能依赖 token ≤ t"和"三角矩阵的解耦结构"是同一件事。

在 PyTorch 里，三角矩阵由两个函数直接生成，名字就来自 **tri**angular：

```python
import torch

T = 5
# torch.tril: 保留下三角（triangular-lower），其余置 0
mask = torch.tril(torch.ones(T, T))
# tensor([[1., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1.]])
```

第 `i` 行的含义是：**query 位置 `i` 允许看哪些 key**。第 0 行只有 1 个 1（只能看自己），第 4 行全是 1（能看前面所有人）。这就是因果性："现在"能看"过去"，不能看"未来"。

---

## 3. 因果掩码：把三角矩阵接进注意力

标准缩放点积注意力（scaled dot-product attention）是：

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```

`QKᵀ` 是一个 `T×T` 的分数矩阵，`scores[i][j]` = query `i` 对 key `j` 的相关度。如果什么都不做，位置 `i` 会看到全部 `j`（包括 `j > i` 的未来），训练时就"作弊"了——模型直接抄答案，学不到真正的语言建模能力。

**因果掩码 = 在 softmax 之前，把上三角（未来）位置加上 `-∞`**：

```python
import torch, math
import torch.nn.functional as F

def causal_attention(q, k, v):
    # q,k,v: (B, n_head, T, head_dim)
    T = q.size(-2)
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)   # (B, nh, T, T)

    # 上三角（不含对角线）设为 -inf —— 这就是因果掩码
    mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)   # -inf 经 softmax 后权重变 0
    return attn @ v
```

关键点：`softmax(-∞) = 0`。被屏蔽的未来位置在加权求和时权重恰好为 0，等价于"不存在"。用 `-∞` 而不是直接把权重设 0，是为了让 softmax 的归一化（分母）也自动排除这些位置，数学上更干净。

### 本仓库怎么做的：一行 `is_causal=True`

CodeChat 没有手写上面那段，而是把整块逻辑交给了 PyTorch 2.x 的融合内核 `scaled_dot_product_attention`（SDPA）。见 `codechat/gpt.py:59-69`：

```python
def forward(self, x):
    B, T, C = x.shape
    qkv = self.qkv(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
    k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
    v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
    # Flash attention via SDPA
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # ← 因果掩码在这里
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.proj(y)
```

`is_causal=True` 这一个参数，替代了"生成三角矩阵 → `masked_fill(-inf)` → softmax"的全过程。这样做有两个实打实的好处，对本仓库的 8B / FSDP 训练至关重要：

1. **省显存**：不必真的物化一个 `T×T`（本仓库 `block_size=2048`，就是 `2048×2048`）的 mask 张量。FlashAttention 类内核在分块（tiling）时**隐式**跳过上三角块，`-∞` 从未在 HBM 里落地。
2. **省算力**：上三角的一半分数**根本不计算**。朴素实现会算满整个矩阵再屏蔽一半，是纯浪费；融合内核直接不算。

这也解释了为什么类名叫 `CausalSelfAttention`（`gpt.py:49`）却在代码里找不到任何 `tril`/`triu`——三角矩阵是概念上的，物理上被内核吸收了。

> 顺带一提，`generate()`（`gpt.py:136-148`）做自回归采样时，每步只取最后一个位置的 logits（`logits[:, -1, :]`）。这本身就是因果性的推理侧体现：预测第 `t+1` 个 token 只依赖前 `t` 个。因果掩码保证了"训练时并行地算 T 个位置"和"推理时逐个生成"这两种模式，学到的是同一个条件分布 `P(x_t | x_<t)`。

---

## 4. 对照：`transformers` 库里的因果掩码

HuggingFace `transformers` 支持从 GPT-2 到 Llama 的一大批模型，还要兼容 padding、左填充、KV-cache、SDPA / eager / FlashAttention 多后端，所以它的因果掩码**不能**只靠 `is_causal=True`——它必须把因果性和 padding mask 合并成一个显式张量。

**经典 GPT-2（`modeling_gpt2.py`）** 的做法最能看清三角矩阵的本质。它在建模时预先注册一个下三角 buffer：

```python
# transformers/models/gpt2/modeling_gpt2.py（精简示意）
self.register_buffer(
    "bias",
    torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool))
        .view(1, 1, max_positions, max_positions),
)

def _attn(self, query, key, value, attention_mask=None):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    attn_weights = attn_weights / (value.size(-1) ** 0.5)

    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    mask_value = torch.finfo(attn_weights.dtype).min      # 该 dtype 能表示的最小值，充当 -inf
    attn_weights = torch.where(causal_mask, attn_weights, mask_value)

    if attention_mask is not None:      # 这是 padding mask，与因果 mask 相加
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, value)
```

对照本仓库，可以看到三点差异：

- **`torch.tril(...)` 是显式的**：GPT-2 把三角矩阵真真切切地物化成一个 buffer，本仓库靠 `is_causal=True` 把它藏进内核。
- **用 `torch.finfo(dtype).min` 而非 `-inf`**：数值稳定，避免 `-inf` 在混合精度下产生 `NaN`。
- **两种 mask 相加**：`causal_mask`（结构）与 `attention_mask`（哪些是 padding，内容）通过加法叠加，一次 softmax 同时满足两个约束。

**现代 Llama 系** 则把这套逻辑抽象成了 `_prepare_4d_causal_attention_mask`（及其新版 `create_causal_mask` / `AttentionMaskConverter`）：它根据 `is_causal`、`sliding_window`、`padding_mask` 动态合成一个 `(B, 1, T, T)` 的 4D float mask，再喂给统一的 attention 接口；当后端是 SDPA 且没有 padding 时，它会**退化成直接传 `is_causal=True`**——绕一圈又回到了本仓库那一行。

结论：**因果掩码的语义在所有实现里完全一致（下三角可见、上三角屏蔽），差异只在"物化 vs. 内核隐式"和"是否要和 padding mask 合并"。** 本仓库因为预训练/SFT 用等长打包（packing），几乎不需要 padding mask，所以能享受最简洁的 `is_causal=True`。

---

## 5. 另一种 mask：本仓库真正显式写出来的 loss mask

回到第 0 节的表格右列。因果掩码在本仓库是隐式的，但**损失掩码是显式且无处不在的**，它用的是 PyTorch 交叉熵的魔数 `-100`。

在模型前向里（`codechat/gpt.py:129-133`）：

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)).float(),
    targets.view(-1),
    ignore_index=-100,          # ← 目标为 -100 的位置不产生任何梯度
)
```

`ignore_index=-100` 是 `nn.CrossEntropyLoss` 的默认值，历史上选 `-100` 是因为它绝不可能与任何合法的类别索引（0..vocab_size-1）冲突，是个安全的"哨兵值"。语义是：**凡是 target == -100 的位置，既不计入 loss，也不回传梯度。**

这在 SFT（监督微调）里是核心机制。看 `codechat/dataloader.py` 的 funcall SFT 分词逻辑（`:208-234`）：

```python
labels: list[int] = []
...
labels.extend([-100] * len(prefix_ids))    # system/user/角色标签 → 不学
...
labels.extend(body_ids + suffix_ids)       # assistant 正文 → 要学
...
labels.extend([-100] * len(span))          # function_response → 不学
# 结尾：若最后一句是 assistant，保留 EOT 让模型学会"停"，否则也 mask
labels.append(EOT if last_role == "assistant" else -100)
```

以及定长打包 / padding 时的补齐（`dataloader.py:79-81`、`:243-244`）：

```python
if len(labels) < self.block_size + 1:
    pad = self.block_size + 1 - len(labels)
    labels = labels + [-100] * pad          # padding 位置 → 不学
```

**为什么要 loss mask？** 一段对话里包含系统提示、用户提问、工具返回、助手回答。我们只想让模型学会"在正确时机生成助手回答（以及何时发起 `<functioncall>`）"，而**不该**去背诵用户的问题或工具的返回值。于是把非助手片段的 label 全设成 `-100`，梯度只从助手 token 流出。这正是 `CLAUDE.md` 里写的那条约定：

> **Loss masking** in funcall SFT: only assistant-segment tokens get gradients; system / user / function_response tokens are `-100`.

**它和因果掩码正交，缺一不可**：因果掩码保证"预测第 t 个 token 时只用前 t-1 个"（结构约束，每条样本都一样）；loss mask 保证"只有助手片段的预测被计入损失"（内容约束，每条样本不同）。一个管**能看见什么**，一个管**要学什么**。

---

## 6. 跑起来看：一个可视化 demo

前面全是文字，容易停留在抽象。仓库里放了一个**能直接跑、会出图**的最小例子：
`docs/examples/causal_mask_demo.py`（只依赖 `torch` + `matplotlib`，无头环境也能跑，
同时在终端打印 ASCII 版）。

```bash
python docs/examples/causal_mask_demo.py
# -> docs/images/causal_mask.png / attention_masked.png / causal_dag.png
```

它用一小段"代码 token" `["def","add","(","a",",","b"]` 演示三件事。

### 6.1 掩码本身就是一个下三角

终端会打印出这张表（行=query，列=key，1=能看见）：

```
        def  add    (    a    ,    b
  def   1    0    0    0    0    0
  add   1    1    0    0    0    0
    (   1    1    1    0    0    0
    a   1    1    1    1    0    0
    ,   1    1    1    1    1    0
    b   1    1    1    1    1    1
```

一眼看出：第 0 行（`def`）只能看自己，最后一行（`b`）能看全部——这就是"现在能看过去、不能看未来"。

### 6.2 掩码如何改变注意力：−∞ → softmax → 0

![掩码前后的注意力](images/attention_masked.png)

三连图从左到右：**①原始分数** `QKᵀ/√d`（满矩阵，上三角也有值）→ **②加掩码**（上三角被置 −∞，图中留白）
→ **③softmax 权重**（上三角恰好变 0，每行归一化后和为 1）。终端同步打印数值，可见未来位置权重全是 `0.00`：

```
       def   add     (     a     ,     b
  add 0.36  0.64  0.00  0.00  0.00  0.00
    ( 0.26  0.65  0.09  0.00  0.00  0.00
```

这就是第 3 节 `softmax(-∞)=0` 那句话的实物。

### 6.3 掩码 ⇔ 一张有向图（DAG）

![自回归 DAG](images/causal_dag.png)

同一件事换个视角：每个 token 的预测**依赖它左边所有 token**，画成有向图就是上面这张——
箭头 `j→i`（`j<i`）密密麻麻地指向右边。这张图的邻接矩阵，正是 6.1 那个下三角。
把它写成概率就是**自回归分解**：

```
P(seq) = P(def|∅)·P(add|def)·P( ( |def,add)·P(a|def,add,()·P( , |…)·P(b|…)
```

**下三角掩码、注意力权重、这张 DAG、这个连乘式——是同一个东西的四种画法。** 这正好把我们引向下一个问题:
这张 DAG 里的"箭头"和因果推断里的"因果箭头",是一回事吗?

---

## 7. 因果推理 ≠ 因果掩码：同名，不同物

用户会很自然地问:贝叶斯网络、有向无环图(DAG)、do-算子那一套"因果推断",
和 Transformer 的"因果掩码"是什么关系?毕竟都叫 **causal**,都画成有向图。

**结论先行:它们共享"有向图 + 拓扑序"这套数学外壳,但"因果"二字的含义完全不同。
Transformer 的 causal mask 里的 "causal" 指的是"时间先后"(temporal / autoregressive),
不是 Judea Pearl 意义上的"因果机制"(mechanism / intervention)。更准确的名字其实是"自回归掩码"。**

### 7.1 两种"因果图"的对照

| | 因果推断的 DAG(Pearl / 贝叶斯网络) | Transformer 的"因果"掩码 |
|---|---|---|
| 节点 | 随机变量(吸烟、焦油、癌症) | 序列位置上的 token |
| 边 `X→Y` | **因果机制**:干预 X 会改变 Y | **时间在先**:位置 j 在 i 左边 |
| 图结构 | 由领域知识/因果发现得到,**稀疏、任意** | **固定的全连下三角**,与内容无关 |
| 核心问题 | `P(Y \| do(X))`——**干预**后会怎样 | `P(x_t \| x_<t)`——**观测**到前文后下一个是什么 |
| 支持反事实 | 是(倒数第三层:若当初不吸烟?) | 否(只建模观测分布) |
| 边的方向可翻转吗 | 不能,翻转就改变因果论断 | 可选(BERT 就用双向,去掉下三角) |

一句话:**因果推断问的是"改变原因,结果会不会变";因果掩码只是保证"算第 t 个词时别偷看后面的词"。**
后者是一条工程约束(防止训练作弊 + 让训练/推理一致),不是关于世界的因果论断。

### 7.2 但它们确实在一个点上接壤:概率的链式法则

联系不是巧合。任何联合分布都能按**任意**变量顺序做链式分解:

```
P(x1,…,xT) = ∏_t P(x_t | x_1,…,x_{t-1})
```

这本身就是一个贝叶斯网络——一个**全连接的下三角 DAG**(第 6.3 节那张图)。
自回归语言模型选定了"从左到右"这一个顺序,并用因果掩码把这个 DAG 的结构**焊死在注意力计算里**:
位置 `i` 的注意力只允许连到 `j≤i`,等价于"节点 i 的父节点 = 它前面所有节点"。

所以这层关系是:**因果掩码 = 一个特定(全连、下三角、按 token 顺序)贝叶斯网络的邻接结构在计算图里的实现。**
它借用了贝叶斯网络的**数学形式**(有向、无环、拓扑序 → 可分解、可并行),
但**没有借用**其**因果语义**(边不代表机制,模型学到的是相关性 `P(next|context)`,而非 `P(·|do(·))`)。

### 7.3 那本项目的大模型里,"因果"到底体现在哪?

回到 CodeChat 自己的代码,把上面的抽象落到实处:

- **结构层面**——因果掩码就是 `codechat/gpt.py:67` 的 `is_causal=True`。它在每一层、每个注意力头里,
  强制实现 7.2 里那个下三角 DAG。模型有 40 层(`8b` preset),但**每一层用的都是同一张下三角图**;
  深度带来的是表示能力,不是更复杂的依赖图——依赖图永远是那个固定的三角。

- **概率层面**——训练目标 `F.cross_entropy(logits, targets)`(`gpt.py:129`)最小化的就是
  `-∑ log P(x_t | x_<t)`,即 7.2 那个连乘式的负对数。模型学的是**观测条件分布**,
  是"给定前文,下一个 token 的经验频率",**不是**"干预某个 token 会如何改变后文"。
  这就是为什么 LLM 会有"相关≠因果"的经典毛病:它没被训练去回答 `do()` 问题。

- **"推理链"是什么?**——当模型做 chain-of-thought、一步步推导时,那条"推理链"看起来很像因果链,
  但它在机制上仍然是**自回归采样**:每一步 `generate()`(`gpt.py:136`)都只是从 `P(x_t|x_<t)` 采样。
  "链"之所以看着有因果性,是因为**训练数据里的人类推理本身有因果结构**,模型把这种结构当作
  统计规律学了进来——它是在**模仿**因果推理的文本表现,而不是在图上跑 do-算子。
  换句话说:**因果掩码提供的是"时间箭头",Pearl 意义上的"因果箭头"(如果有的话)是从数据里学来的软知识,
  不由掩码保证、也不受掩码约束。**

- **反过来印证**——funcall 场景(`codechat/funcall_reward.py`、`dataloader.py` 的 loss mask)也一样:
  模型学会"用户问天气 → 发 `<functioncall>`",这是数据里的**条件相关**被 SFT 固化,
  不是模型"理解"了调用天气 API 会导致得到天气。它不需要理解因果,只需要拟合 `P(下一段 | 前文)`。

### 7.4 一句话记住

> **贝叶斯网络的 DAG 用有向边编码"谁导致谁"(机制),回答干预与反事实;
> Transformer 的因果掩码用下三角编码"谁在谁之前"(时间),只回答"下一个词是什么"。
> 二者共享"有向无环 + 拓扑序 → 可分解"的数学骨架,这也是"因果掩码"这个名字的来处——
> 但它建模的是概率的链式法则,不是 Pearl 的因果阶梯。本项目的大模型自始至终工作在观测分布 `P(x_t|x_<t)` 上,
> "因果掩码"给的是时间序,不是因果性。**

---

## 8. 小结

| 概念 | 起源 | 数学形态 | 本仓库位置 |
|---|---|---|---|
| **mask（广义）** | 信号处理 / 位运算的"遮片"隐喻 | 逐元素 0/1 或 0/−∞ | 贯穿全仓 |
| **三角矩阵** | 19 世纪线性代数（LU/Cholesky/前代回代） | 下三角 `tril` / 上三角 `triu` | 概念上存在，被 SDPA 内核吸收 |
| **因果掩码** | 《Attention Is All You Need》(2017) 的 masked self-attention | 下三角可见、上三角置 −∞ | `is_causal=True`（`gpt.py:67`） |
| **loss mask** | PyTorch `CrossEntropyLoss` 的 `ignore_index=-100` | 一维哨兵向量 | `-100`（`gpt.py:132`、`dataloader.py`） |

三条线索最终汇成一句话：

> **自回归语言模型 = 用一个"下三角形状"的约束，把"未来不可见"这条因果律写进注意力（因果掩码 / 三角矩阵）；再用一个"哨兵向量"的约束，把"哪些 token 值得学"写进损失（loss mask）。**
> 前者在本仓库被 `is_causal=True` 一行浓缩，后者被 `-100` 显式铺陈——理解它们的分工，就理解了从 `codechat/gpt.py` 到 `codechat/dataloader.py` 整条训练管线的骨架。

### 参考

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
- PyTorch 文档：`torch.nn.functional.scaled_dot_product_attention`、`torch.tril` / `torch.triu`、`torch.nn.CrossEntropyLoss`。
- HuggingFace `transformers`：`models/gpt2/modeling_gpt2.py`、`modeling_attn_mask_utils.py`（`AttentionMaskConverter` / `create_causal_mask`）。
- 本仓库：`codechat/gpt.py`、`codechat/dataloader.py`、`CLAUDE.md`（loss-masking 约定）。
