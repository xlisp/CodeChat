import os
import torch
import torch.distributed as dist


def _is_fsdp(model) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        return isinstance(model, FSDP)
    except Exception:
        return False


def _unwrap(model):
    # DDP / FSDP both expose .module, but for a vanilla nn.Module we return as-is.
    return getattr(model, "module", model)


def save(path: str, model, optimizer, step: int, cfg):
    """Save a full (un-sharded) checkpoint on rank 0.

    Works for:
      - single-GPU training (model is a plain nn.Module)
      - DDP (model.module is unwrapped transparently)
      - FSDP (full state_dict is gathered onto rank 0 via summon_full_params)
    """
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state = model.state_dict()
    else:
        model_state = _unwrap(model).state_dict()

    # Only rank 0 actually writes the file.
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank != 0:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model_state,
        # Optimizer state for FSDP would need its own gather; we skip it for simplicity
        # (resume from pretrain typically restarts the optimizer anyway).
        "optimizer": optimizer.state_dict() if (optimizer is not None and not _is_fsdp(model)) else None,
        "step": step,
        "cfg": cfg.__dict__,
    }
    torch.save(payload, path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def load(path: str, map_location="cuda"):
    return torch.load(path, map_location=map_location, weights_only=False)
