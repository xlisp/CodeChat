"""可跑的因果掩码科普 demo。

运行:
    python docs/examples/causal_mask_demo.py

产出（写到 docs/images/）:
    1. causal_mask.png        —— 下三角掩码本身
    2. attention_masked.png   —— 原始分数 / 掩码后 / softmax 三连图
    3. causal_dag.png         —— 自回归分解对应的贝叶斯网络（DAG）

无显示环境也能跑（用 Agg 后端），且全程在终端打印 ASCII 版本，
所以就算没装 matplotlib，删掉画图部分也能看懂。
"""
import os
import math
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # 无头环境
import matplotlib.pyplot as plt

# 让中文能正常显示（按可用性回退；找不到就退回 DejaVu，中文会变方框但不报错）
matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "Hiragino Sans GB", "STHeiti", "Heiti TC",
    "PingFang HK", "Songti SC", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.abspath(os.path.join(HERE, "..", "images"))
os.makedirs(IMG, exist_ok=True)

TOKENS = ["def", "add", "(", "a", ",", "b"]   # 一小段"代码"当例子
T = len(TOKENS)


# ---------------------------------------------------------------------------
# 0) 给小白的"座位表"：谁能看谁（配文章《写在前面·场景三》）
#    例子就用 "我爱吃苹果"，与正文 ASCII 表一一对应。
# ---------------------------------------------------------------------------
def demo_seatchart():
    words = ["我", "爱", "吃", "苹果"]
    n = len(words)
    visible = torch.tril(torch.ones(n, n))  # 1=能看, 0=不能看

    plt.figure(figsize=(4.6, 4.2))
    # 绿=能看(1)，灰=不能看(0)
    plt.imshow(visible, cmap="Greens", vmin=-0.3, vmax=1)
    plt.xticks(range(n), words, fontsize=13)
    plt.yticks(range(n), words, fontsize=13)
    plt.xlabel("能不能看见这个字（列）", fontsize=11)
    plt.ylabel("正在预测这个字（行）", fontsize=11)
    plt.title("座位表：谁能看谁  绿=能看  灰=不能看", fontsize=12)
    for i in range(n):
        for j in range(n):
            can = int(visible[i, j])
            plt.text(j, i, "能看" if can else "遮住",
                     ha="center", va="center", fontsize=11,
                     color="white" if can else "dimgrey")
    plt.tight_layout()
    p = os.path.join(IMG, "seat_chart.png")
    plt.savefig(p, dpi=130); plt.close()
    print("\n[0] 座位表 (行=预测的字, 列=能不能看见):")
    print("        " + "  ".join(f"{w:>3}" for w in words))
    for i, w in enumerate(words):
        row = "  ".join(("能看" if v else "遮住") for v in visible[i])
        print(f"  {w:>4} {row}")
    print(f"    -> saved {p}")


# ---------------------------------------------------------------------------
# 1) 因果掩码本身：一个下三角矩阵
# ---------------------------------------------------------------------------
def demo_mask():
    # 与 torch.tril(ones) 等价；1=可见, 0=屏蔽
    visible = torch.tril(torch.ones(T, T))
    print("\n[1] 因果掩码 (1=能看见, 0=被屏蔽)  行=query 位置, 列=key 位置")
    print("        " + "  ".join(f"{t:>3}" for t in TOKENS))
    for i, t in enumerate(TOKENS):
        row = "  ".join(f"{int(v):>3}" for v in visible[i])
        print(f"  {t:>5} {row}")

    plt.figure(figsize=(4.2, 3.8))
    plt.imshow(visible, cmap="Greens", vmin=0, vmax=1)
    plt.xticks(range(T), TOKENS)
    plt.yticks(range(T), TOKENS)
    plt.xlabel("key (token being looked at)")
    plt.ylabel("query (token being predicted)")
    plt.title("Causal mask = torch.tril(ones)")
    for i in range(T):
        for j in range(T):
            plt.text(j, i, int(visible[i, j]), ha="center", va="center",
                     color="white" if visible[i, j] else "grey", fontsize=9)
    plt.tight_layout()
    p = os.path.join(IMG, "causal_mask.png")
    plt.savefig(p, dpi=130); plt.close()
    print(f"    -> saved {p}")


# ---------------------------------------------------------------------------
# 2) 掩码如何改变注意力：原始分数 -> 加 -inf -> softmax
# ---------------------------------------------------------------------------
def demo_attention():
    torch.manual_seed(0)
    d_k = 8
    q = torch.randn(T, d_k)
    k = torch.randn(T, d_k)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)      # (T, T)

    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()   # 上三角=未来
    masked = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(masked, dim=-1)                         # -inf -> 权重 0

    print("\n[2] softmax 后的注意力权重（每行和为 1，未来位置=0.00）")
    print("        " + "  ".join(f"{t:>4}" for t in TOKENS))
    for i, t in enumerate(TOKENS):
        row = "  ".join(f"{v:0.2f}" for v in attn[i])
        print(f"  {t:>5} {row}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, mat, title, cmap in [
        (axes[0], scores, "1) raw scores  QKt/sqrt(d)", "coolwarm"),
        (axes[1], masked.masked_fill(mask, float("nan")), "2) masked (upper tri = -inf)", "coolwarm"),
        (axes[2], attn, "3) softmax weights", "viridis"),
    ]:
        im = ax.imshow(mat, cmap=cmap)
        ax.set_xticks(range(T)); ax.set_xticklabels(TOKENS, rotation=45, ha="right")
        ax.set_yticks(range(T)); ax.set_yticklabels(TOKENS)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    p = os.path.join(IMG, "attention_masked.png")
    plt.savefig(p, dpi=130); plt.close()
    print(f"    -> saved {p}")


# ---------------------------------------------------------------------------
# 3) 因果掩码 <=> 一个 DAG（贝叶斯网络的自回归分解）
#    P(x1..xT) = ∏ P(x_t | x_<t)：每个 token 的父节点 = 它左边所有 token
# ---------------------------------------------------------------------------
def demo_dag():
    print("\n[3] 自回归分解 = 一个下三角 DAG:")
    print("    P(seq) = " + " · ".join(
        f"P({t}|{','.join(TOKENS[:i]) or '∅'})" for i, t in enumerate(TOKENS)))

    plt.figure(figsize=(8, 2.6))
    xs = list(range(T))
    ys = [0] * T
    # 边：j -> i 对 j < i（每个节点依赖它左边所有节点）
    for i in range(T):
        for j in range(i):
            # 弧线，越远的依赖画得越高，避免重叠
            rad = 0.15 + 0.12 * (i - j)
            plt.annotate("", xy=(i, 0), xytext=(j, 0),
                         arrowprops=dict(arrowstyle="->", color="tab:blue",
                                         alpha=0.5,
                                         connectionstyle=f"arc3,rad={rad}"))
    plt.scatter(xs, ys, s=1400, c="white", edgecolors="black", zorder=3)
    for x, t in zip(xs, TOKENS):
        plt.text(x, 0, t, ha="center", va="center", zorder=4, fontsize=11)
    plt.title("Autoregressive DAG: edge j->i means predicting token i depends on token j (j<i)")
    plt.axis("off")
    plt.xlim(-0.6, T - 0.4); plt.ylim(-1.2, 1.4)
    plt.tight_layout()
    p = os.path.join(IMG, "causal_dag.png")
    plt.savefig(p, dpi=130); plt.close()
    print(f"    -> saved {p}")


if __name__ == "__main__":
    demo_seatchart()
    demo_mask()
    demo_attention()
    demo_dag()
    print("\n完成。三张图在 docs/images/ 下。")
