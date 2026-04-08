"""Optimizer factory. We use AdamW for everything by default to keep it simple.

(nanochat uses Muon for the 2D weights and AdamW for embeddings/head; for a
single-GPU CodeChat training, plain AdamW is entirely sufficient and simpler.)
"""
import torch


def build_optimizer(model: torch.nn.Module, lr: float = 3e-4, weight_decay: float = 0.1, betas=(0.9, 0.95)):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=torch.cuda.is_available())


def cosine_lr(step: int, max_steps: int, lr: float, warmup: int = 200, min_ratio: float = 0.1) -> float:
    import math
    if step < warmup:
        return lr * step / max(1, warmup)
    if step >= max_steps:
        return lr * min_ratio
    progress = (step - warmup) / max(1, max_steps - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
