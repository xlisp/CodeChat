"""Minimal GPT transformer. Close to nanochat / nanoGPT style.

One dial of complexity: `depth`. Width/heads are derived from it so that the
model stays roughly compute-optimal and behaves well on a single A800 80GB.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_fn

from .common import COMPUTE_DTYPE


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    depth: int = 32           # number of transformer blocks
    n_embd: int = 2560        # hidden size
    n_head: int = 20          # attention heads (head_dim = n_embd // n_head = 128)
    block_size: int = 2048    # context length
    dropout: float = 0.0
    grad_checkpoint: bool = True   # activation checkpointing to fit 2B on 80GB


# Preset: ~2B parameters. Roughly 12*L*d^2 + vocab*d
#   depth=32, n_embd=2560  ->  12*32*2560^2 ≈ 2.52B  (+ 128M tied embeddings)
#   actual count including heads/norms ≈ 2.1–2.2B
PRESETS = {
    "d20":  dict(depth=20, n_embd=1280, n_head=10),   # ~400M
    "d24":  dict(depth=24, n_embd=1792, n_head=14),   # ~900M
    "2b":   dict(depth=32, n_embd=2560, n_head=20),   # ~2.1B
    "3b":   dict(depth=32, n_embd=3072, n_head=24),   # ~3.0B (tight on 80GB)
}


def make_config(preset: str = "2b", **overrides) -> "GPTConfig":
    p = dict(PRESETS[preset])
    p.update(overrides)
    return GPTConfig(**p)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Flash attention via SDPA
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd, dtype=COMPUTE_DTYPE)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd, dtype=COMPUTE_DTYPE)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.ln_f = nn.RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"seq len {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            if self.cfg.grad_checkpoint and self.training:
                x = ckpt_fn(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            targets.view(-1),
            ignore_index=-100,
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, top_k: int | None = 50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float() / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
