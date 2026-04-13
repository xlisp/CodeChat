"""Supervised fine-tuning on Python instruction data.

Supports both single-GPU and multi-GPU (FSDP) training.  Launch mode is
auto-detected from torchrun env vars, identical to base_train.py.

  Single GPU:  python -m scripts.chat_sft --base-ckpt ...
  8x A800:     torchrun --nproc_per_node=8 -m scripts.chat_sft --base-ckpt ...
"""
import argparse
import functools
import os
import time
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from codechat.common import COMPUTE_DTYPE, seed_all
from codechat.gpt import GPT, GPTConfig, Block
from codechat.dataloader import SFTLoader
from codechat.optim import build_optimizer, cosine_lr
from codechat.checkpoint import save as save_ckpt, load as load_ckpt


def setup_distributed():
    """Detect torchrun and initialise NCCL. Returns (is_dist, local_rank, rank, world_size)."""
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, local_rank, dist.get_rank(), dist.get_world_size()


def wrap_fsdp(model):
    """Wrap the model with FSDP, sharding at the transformer Block boundary."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, BackwardPrefetch
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block},
    )
    mp_policy = MixedPrecision(
        param_dtype=COMPUTE_DTYPE,
        reduce_dtype=COMPUTE_DTYPE,
        buffer_dtype=COMPUTE_DTYPE,
    )
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        limit_all_gathers=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--data-dir", default="data/sft")
    ap.add_argument("--device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=100)
    # see scripts/base_train.py for why --run-name is the primary form
    ap.add_argument("--run-name", "--run", dest="run", default="codechat_d20_sft")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--tb-dir", default="runs/tb", help="tensorboard log root")
    args = ap.parse_args()

    is_dist, local_rank, rank, world_size = setup_distributed()
    is_master = (rank == 0)

    seed_all(1337 + rank)
    device = torch.device("cuda", local_rank) if is_dist else torch.device("cuda")
    assert device.type == "cuda"
    torch.set_float32_matmul_precision("high")

    # Load checkpoint to CPU so all ranks don't compete for GPU 0 memory.
    state = load_ckpt(args.base_ckpt, map_location="cpu")
    cfg = GPTConfig(**state["cfg"])
    model = GPT(cfg)
    model.load_state_dict(state["model"])
    del state  # free CPU memory
    if is_master:
        print(f"loaded base ckpt {args.base_ckpt}")

    model = model.to(device).to(COMPUTE_DTYPE)
    if is_dist:
        model = wrap_fsdp(model)

    loader = SFTLoader(os.path.join(args.data_dir, "train.jsonl"),
                       args.device_batch_size, cfg.block_size, device)
    optim = build_optimizer(model, lr=args.lr)

    ckpt_path = os.path.join(args.ckpt_dir, args.run, "latest.pt")
    tb_path = os.path.join(args.tb_dir, args.run)
    writer = SummaryWriter(log_dir=tb_path) if is_master else None
    if is_master:
        print(f"tensorboard -> {tb_path}")
    t0 = time.time()
    model.train()
    for step in range(1, args.max_steps + 1):
        lr = cosine_lr(step, args.max_steps, args.lr, warmup=args.warmup)
        for g in optim.param_groups:
            g["lr"] = lr
        optim.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(args.grad_accum):
            x, y = loader.next_batch()
            if is_dist and micro < args.grad_accum - 1 and hasattr(model, "no_sync"):
                with model.no_sync():
                    _, loss = model(x, y)
                    (loss / args.grad_accum).backward()
            else:
                _, loss = model(x, y)
                (loss / args.grad_accum).backward()
            loss_accum += loss.item() / args.grad_accum

        if is_dist and hasattr(model, "clip_grad_norm_"):
            grad_norm = model.clip_grad_norm_(1.0)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if is_master:
            writer.add_scalar("sft/loss", loss_accum, step)
            writer.add_scalar("sft/lr", lr, step)
            writer.add_scalar("sft/grad_norm", float(grad_norm), step)
            writer.add_scalar("sft/elapsed_s", time.time() - t0, step)

        if step % 20 == 0 and is_master:
            print(f"sft step {step:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | {time.time()-t0:.0f}s")
        if step % 500 == 0 or step == args.max_steps:
            save_ckpt(ckpt_path, model, optim, step, cfg)
            if is_master:
                print(f"  saved -> {ckpt_path}")
    if is_master:
        writer.close()
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
