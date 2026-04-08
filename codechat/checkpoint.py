import os
import torch


def save(path: str, model, optimizer, step: int, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "cfg": cfg.__dict__,
        },
        path,
    )


def load(path: str, map_location="cuda"):
    return torch.load(path, map_location=map_location, weights_only=False)
