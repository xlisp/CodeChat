"""可跑的 demo：从点积到答案，以及 RL 的策略梯度到底改了什么。

运行:
    python docs/examples/rl_inference_viz.py

产出（写到 docs/images/）:
    1. viz_dotprod_to_answer.png  —— 一条 prompt 从"点积"走到"采样出答案"的全链路
    2. viz_policy_gradient_shift.png —— REINFORCE 把概率质量从坏答案搬到好答案的过程
    3. viz_param_subspace.png     —— 一次 RL 更新的 ΔW 是低秩的（语义拟合的参数子空间）

配文：docs/sft_rl_inference_mechanics.md 第 9、10 节。

说明：
  * 图 2 的 reward 用的是**本仓库真实代码** codechat/funcall_reward.py 的
    阶梯奖励，候选串就是模型可能吐出的 6 种输出；策略更新用的是
    scripts/chat_rl_funcall.py:375-405 一模一样的公式
    （advantages = r - r.mean(); loss = -(A * logp).mean()）。
  * 图 1、图 3 用的是**玩具模型**（8 维 embedding、32 词表），只为把
    gpt.py 里真实发生的矩阵运算画出来，数值本身没有语义含义。
无显示环境也能跑（Agg 后端），全程在终端打印 ASCII 版本。
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

import sys
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from codechat.funcall_reward import funcall_reward  # noqa: E402


# ---------------------------------------------------------------------------
# 1) 点积 -> 注意力 -> 残差流 -> 词表打分 -> 采样
#    对应 gpt.py:67 (SDPA)、gpt.py:126 (head)、gpt.py:141-146 (采样)
# ---------------------------------------------------------------------------
PROMPT = ["<|system|>", "get_weather", "<|user|>", "weather", "in", "Tokyo", "<|assistant|>"]
# 玩具词表：前 6 个是 funcall 相关，后 6 个是写代码相关，剩下是杂项
VOCAB = ["<functioncall>", "{", '"name"', "get_weather", "Tokyo", "<|end|>",
         "def", "quicksort", "return", "for", "sorted", "list",
         "the", "a", "is", "and"]
D = 8


def _toy_embeddings(seed=0):
    """构造两簇有结构的 embedding：funcall 簇 / code 簇，簇内点积大、跨簇点积小。

    真实模型里这个结构是预训练+SFT 学出来的；这里手工造出来，只为看清
    "点积 = 语义相似度打分" 这件事。
    """
    g = torch.Generator().manual_seed(seed)
    E = torch.randn(len(VOCAB), D, generator=g) * 0.25
    funcall_dir = torch.zeros(D); funcall_dir[0] = 1.0
    code_dir = torch.zeros(D); code_dir[1] = 1.0
    for i in range(len(VOCAB)):
        E[i] += funcall_dir * 1.2 if i < 6 else code_dir * 1.2
    return F.normalize(E, dim=-1)


def demo_dotprod_to_answer():
    E = _toy_embeddings()                       # 词嵌入 = gpt.py:100 的 tok_emb.weight
    idx = [VOCAB.index(t) if t in VOCAB else 0 for t in PROMPT]
    # prompt 里有词表外的词，用最近的簇代替（玩具处理）
    idx = [0, 3, 12, 12, 13, 4, 0]
    T = len(idx)
    x = E[idx]                                   # [T, D]  gpt.py:119

    # --- 第一种点积：QK^T，token 之间互相打分 ---
    q = k = x                                    # 玩具：直接拿 embedding 当 Q/K
    scores = (q @ k.T) / math.sqrt(D)
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    attn = F.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)   # gpt.py:67

    # --- 残差流：最后一个位置的隐状态 = 注意力加权和 ---
    h = attn[-1] @ x                             # [D]
    h = F.normalize(h, dim=-1)                   # 代替 ln_f，gpt.py:125

    # --- 第二种点积：h · E^T，拿隐状态去和整个词表打分（权重绑定：head=tok_emb） ---
    logits = (h @ E.T) * 6.0                     # gpt.py:126，乘 6 只为拉开差距
    top_k = 5
    v, _ = torch.topk(logits, top_k)
    kept = logits.clone()
    kept[kept < v[-1]] = -float("inf")           # gpt.py:143-144
    probs = F.softmax(kept / 0.7, dim=-1)        # temperature=0.7，gpt.py:141

    print("\n[1] 点积 -> 答案 全链路")
    print("    prompt:", " ".join(PROMPT))
    print("    最后一个位置的注意力权重（它在 '看' 谁）:")
    for t, w in zip(PROMPT, attn[-1]):
        bar = "█" * int(w * 40)
        print(f"      {t:>20}  {w:0.3f} {bar}")
    print("    词表打分 top-5（h·E^T 的结果）:")
    order = torch.argsort(logits, descending=True)[:top_k]
    for i in order:
        print(f"      {VOCAB[i]:>16}  logit {logits[i]:+6.2f}   p={probs[i]:0.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))

    im = axes[0].imshow(scores, cmap="coolwarm")
    axes[0].set_title("① QK^T/√d：token 两两点积\n(gpt.py:67 SDPA 内部)", fontsize=10)
    axes[0].set_xticks(range(T)); axes[0].set_xticklabels(PROMPT, rotation=60, ha="right", fontsize=7)
    axes[0].set_yticks(range(T)); axes[0].set_yticklabels(PROMPT, fontsize=7)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    im = axes[1].imshow(attn, cmap="viridis")
    axes[1].set_title("② 因果掩码 + softmax\n= 注意力权重", fontsize=10)
    axes[1].set_xticks(range(T)); axes[1].set_xticklabels(PROMPT, rotation=60, ha="right", fontsize=7)
    axes[1].set_yticks(range(T)); axes[1].set_yticklabels(PROMPT, fontsize=7)
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].barh(range(len(VOCAB)), logits.tolist(),
                 color=["tab:orange" if i < 6 else "tab:blue" for i in range(len(VOCAB))])
    axes[2].set_yticks(range(len(VOCAB))); axes[2].set_yticklabels(VOCAB, fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set_title("③ h·E^T：隐状态和每个词点积\n(gpt.py:126，head 与 tok_emb 权重绑定)", fontsize=10)
    axes[2].set_xlabel("logit")

    axes[3].barh(range(len(VOCAB)), probs.tolist(),
                 color=["tab:orange" if i < 6 else "tab:blue" for i in range(len(VOCAB))])
    axes[3].set_yticks(range(len(VOCAB))); axes[3].set_yticklabels(VOCAB, fontsize=8)
    axes[3].invert_yaxis()
    axes[3].set_title("④ top-k=5 截断 + T=0.7 softmax\n-> multinomial 采样 (gpt.py:141-146)", fontsize=10)
    axes[3].set_xlabel("采样概率")

    plt.tight_layout()
    p = os.path.join(IMG, "viz_dotprod_to_answer.png")
    plt.savefig(p, dpi=130); plt.close()
    print(f"    -> saved {p}")


# ---------------------------------------------------------------------------
# 2) 策略梯度：用真实的 funcall_reward 阶梯，跑 chat_rl_funcall.py 的更新公式
# ---------------------------------------------------------------------------
GT_NAME = "get_weather"
GT_ARGS = {"location": "Tokyo", "unit": "celsius"}

CANDIDATES = [
    ("纯文本(无 tag)",
     "Sure! Here is quicksort in Python:\ndef quicksort(a): ..."),
    ("tag+坏 JSON",
     "<functioncall> {name: get_weather, arguments: }"),
    ("缺 name",
     '<functioncall> {"arguments": {"location": "Tokyo", "unit": "celsius"}}'),
    ("名字错",
     '<functioncall> {"name": "get_time", "arguments": {"location": "Tokyo"}}'),
    ("名字对/参数半对",
     '<functioncall> {"name": "get_weather", "arguments": {"location": "Tokyo"}}'),
    ("完全正确",
     '<functioncall> {"name": "get_weather", '
     '"arguments": {"location": "Tokyo", "unit": "celsius"}}'),
]


def demo_policy_gradient():
    rewards_of = []
    tiers = []
    for _, text in CANDIDATES:
        r, tier = funcall_reward(text, GT_NAME, GT_ARGS)
        rewards_of.append(r); tiers.append(tier)
    R = torch.tensor(rewards_of)

    print("\n[2] 真实 funcall_reward 给 6 种输出的打分:")
    for (name, _), r, t in zip(CANDIDATES, rewards_of, tiers):
        print(f"      {name:>16}  reward={r:0.3f}  tier={t}")

    # 一个只有 6 个动作的"策略"：logits 就是可训练参数 θ。
    # 初始分布模仿一个 SFT 完但还不精确的模型：会输出 tag，但参数常填错。
    theta = torch.tensor([1.0, 0.6, 0.4, 1.2, 1.4, 0.2], requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.05)
    K = 16                      # = chat_rl_funcall.py 的 --num-samples
    steps = 200
    g = torch.Generator().manual_seed(1337)

    prob_hist, rmean_hist, rstd_hist, advabs_hist = [], [], [], []
    theta0 = theta.detach().clone()
    for step in range(steps):
        probs = F.softmax(theta, dim=-1)
        prob_hist.append(probs.detach().clone())
        # --- rollouts：从当前策略采 K 个样本（chat_rl_funcall.sample_batch）---
        a = torch.multinomial(probs.detach(), K, replacement=True, generator=g)
        r = R[a]
        # --- REINFORCE with baseline，chat_rl_funcall.py:375 ---
        adv = r - r.mean()
        logp = torch.log(probs[a] + 1e-9)
        pg_loss = -(adv * logp).mean()          # chat_rl_funcall.py:404
        opt.zero_grad(); pg_loss.backward(); opt.step()

        rmean_hist.append(r.mean().item())
        rstd_hist.append(r.std().item())
        advabs_hist.append(adv.abs().mean().item())

    P = torch.stack(prob_hist)                  # [steps, 6]
    dtheta = (theta.detach() - theta0)

    print("    RL 前后的动作概率:")
    for i, (name, _) in enumerate(CANDIDATES):
        print(f"      {name:>16}  {P[0, i]:0.3f} -> {P[-1, i]:0.3f}   "
              f"Δlogit={dtheta[i]:+0.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    colors = plt.cm.RdYlGn(torch.linspace(0.1, 0.9, len(CANDIDATES)).numpy())
    for i, (name, _) in enumerate(CANDIDATES):
        axes[0].plot(P[:, i], label=f"{name} (r={rewards_of[i]:.2f})", color=colors[i], lw=2)
    axes[0].set_title("① 概率质量搬家：RL 每一步都在重排 logits", fontsize=11)
    axes[0].set_xlabel("RL step"); axes[0].set_ylabel("该输出被采样的概率")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].barh(range(len(CANDIDATES)), dtheta.tolist(), color=colors)
    axes[1].set_yticks(range(len(CANDIDATES)))
    axes[1].set_yticklabels([n for n, _ in CANDIDATES], fontsize=9)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_title("② 策略梯度写进参数的东西：Δlogit\n(推理时唯一的变化就是这个)", fontsize=11)
    axes[1].set_xlabel("θ_RL − θ_SFT")

    axes[2].plot(rmean_hist, label="reward_mean", color="tab:green")
    axes[2].plot(rstd_hist, label="reward_std", color="tab:orange")
    axes[2].plot(advabs_hist, label="|advantage| mean", color="tab:red")
    axes[2].set_title("③ 饱和：reward_std→0 则 advantage→0，梯度消失\n"
                      "(v6 报告里 step 120/210/270 的 std=0.000 就是这个)", fontsize=11)
    axes[2].set_xlabel("RL step"); axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(IMG, "viz_policy_gradient_shift.png")
    plt.savefig(p, dpi=130); plt.close()
    print(f"    -> saved {p}")


# ---------------------------------------------------------------------------
# 3) 参数子空间：一次 RL 更新的 ΔW 是低秩的
#    pg_loss 对 head 权重的梯度 = 每个 (rollout, 位置) 贡献一个秩-1 外积，
#    所以 rank(ΔW) <= K * T « d。SFT 一次 step 也是同理，只是 K*T 大得多。
# ---------------------------------------------------------------------------
def _grad_of_head(W, h, tgt, adv=None):
    """一次更新对 head 权重的梯度。adv=None 时是 SFT 的交叉熵，否则是 REINFORCE。"""
    if W.grad is not None:
        W.grad = None
    logits = h @ W.T                         # gpt.py:126 的同一次点积
    if adv is None:
        loss = F.cross_entropy(logits.reshape(-1, W.shape[0]).float(), tgt.reshape(-1))
    else:
        logp = F.log_softmax(logits.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        loss = -(adv * logp.mean(dim=1)).mean()    # chat_rl_funcall.py:402-404
    loss.backward()
    return W.grad.clone()


def demo_param_subspace():
    torch.manual_seed(0)
    d, V = 256, 512          # 玩具版的 n_embd / vocab_size
    W = torch.randn(V, d, requires_grad=True) * 0.02
    W = W.detach().requires_grad_(True)

    # RL 一步：K=16 个 rollout，每个只有 ~8 个被监督的 token（funcall JSON 很短，
    # 且 chat_rl_funcall.py:393 的 keep_mask 把 EOT 之后全部丢掉）
    K, T_rl = 16, 8
    h_rl = torch.randn(K, T_rl, d) * 0.5
    tgt_rl = torch.randint(0, V, (K, T_rl))
    adv = torch.randn(K) * 0.3
    gW_rl = _grad_of_head(W, h_rl, tgt_rl, adv)

    # SFT 一步：device_batch_size × block_size × grad_accum，token 数比 RL 多两个量级
    B, T_sft = 64, 32
    h_sft = torch.randn(B, T_sft, d) * 0.5
    tgt_sft = torch.randint(0, V, (B, T_sft))
    gW_sft = _grad_of_head(W, h_sft, tgt_sft, None)

    s_rl = torch.linalg.svdvals(gW_rl)
    s_sft = torch.linalg.svdvals(gW_sft)
    tol = 1e-6
    eff_rl = int((s_rl > s_rl[0] * tol).sum())
    eff_sft = int((s_sft > s_sft[0] * tol).sum())
    # 能量集中度：前 n 个奇异方向装了多少比例的 ‖ΔW‖²
    e_rl = torch.cumsum(s_rl ** 2, 0) / (s_rl ** 2).sum()
    e_sft = torch.cumsum(s_sft ** 2, 0) / (s_sft ** 2).sum()
    n90_rl = int((e_rl < 0.9).sum()) + 1
    n90_sft = int((e_sft < 0.9).sum()) + 1

    print("\n[3] 一次更新的 ΔW 住在多大的子空间里")
    print(f"      矩阵形状 = ({V}, {d})，满秩 = {min(V, d)}")
    print(f"      RL  一步:  秩 = {eff_rl}  (上界 K×T = {K*T_rl})，"
          f"90% 能量落在前 {n90_rl} 个方向")
    print(f"      SFT 一步:  秩 = {eff_sft}  (上界 B×T = {B*T_sft})，"
          f"90% 能量落在前 {n90_sft} 个方向")
    print(f"      ‖ΔW_RL‖_F = {gW_rl.norm():.4e}   ‖ΔW_SFT‖_F = {gW_sft.norm():.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].semilogy((s_rl / s_rl[0]).detach(), color="tab:red",
                     label=f"RL 一步：秩={eff_rl} (K×T={K*T_rl})")
    axes[0].semilogy((s_sft / s_sft[0]).detach(), color="tab:blue",
                     label=f"SFT 一步：秩={eff_sft} (B×T={B*T_sft})")
    axes[0].axvline(K * T_rl, ls="--", color="grey", lw=1)
    axes[0].text(K * T_rl + 4, 1e-5, "K×T", color="grey", fontsize=9)
    axes[0].set_ylim(1e-8, 2)
    axes[0].set_title("① 奇异值谱：ΔW 的秩 ≤ 这一步用到的 token 数\n"
                      "RL 一步只碰 K×T 个 token，所以掉进低维子空间", fontsize=11)
    axes[0].set_xlabel("奇异值序号 i"); axes[0].set_ylabel("σ_i / σ_0（对数）")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].plot(e_rl.detach(), color="tab:red", label=f"RL：90% 能量在前 {n90_rl} 维")
    axes[1].plot(e_sft.detach(), color="tab:blue", label=f"SFT：90% 能量在前 {n90_sft} 维")
    axes[1].axhline(0.9, ls="--", color="grey", lw=1)
    axes[1].set_title("② 累计能量：这一步的位移方向有多集中", fontsize=11)
    axes[1].set_xlabel("前 i 个奇异方向"); axes[1].set_ylabel("累计 σ² 占比")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(IMG, "viz_param_subspace.png")
    plt.savefig(p, dpi=130); plt.close()
    print(f"    -> saved {p}")


if __name__ == "__main__":
    demo_dotprod_to_answer()
    demo_policy_gradient()
    demo_param_subspace()
    print("\n完成。三张图在 docs/images/ 下。")
