"""Global utilities: device, dtype, seeding. nanochat-style explicit precision."""
import os
import random
import numpy as np
import torch

# Single global compute dtype. A800 supports native bf16 tensor cores.
_DTYPE_STR = os.environ.get("CODECHAT_DTYPE", "bfloat16")
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
COMPUTE_DTYPE = _DTYPE_MAP[_DTYPE_STR]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
