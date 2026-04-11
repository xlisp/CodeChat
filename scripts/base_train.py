"""CodeChat pretraining.

Runs on:
  - a single A800 80GB (preset=2b)  ->  `python -m scripts.base_train ...`
  - 8x A800 80GB via FSDP (preset=8b) ->
        torchrun --nproc_per_node=8 -m scripts.base_train ...

Launch mode is auto-detected from the torchrun env vars (`LOCAL_RANK` /
`RANK` / `WORLD_SIZE`). In distributed mode the model is wrapped with FSDP
(FULL_SHARD) so params, gradients and AdamW states are sharded across ranks
— this is mandatory for the 8B preset because a full fp32 AdamW state alone
is ~96GB, far beyond a single 80GB card.
"""
import argparse
import os
import time
import functools
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from codechat.common import DEVICE, COMPUTE_DTYPE, seed_all, get_num_params
from codechat.gpt import GPT, GPTConfig, Block, make_config, PRESETS
from codechat.dataloader import PretrainLoader
from codechat.optim import build_optimizer, cosine_lr
from codechat.checkpoint import save as save_ckpt, load as load_ckpt
from codechat.tokenizer import VOCAB_SIZE


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
    ap.add_argument("--data-dir", default="data/pretrain")
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="2b")
    ap.add_argument("--block-size", type=int, default=2048)
    ap.add_argument("--device-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--run", default="codechat_2b")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--tb-dir", default="runs/tb", help="tensorboard log root")
    ap.add_argument("--resume", default=None, help="optional checkpoint to warm-start from (model weights only)")
    args = ap.parse_args()

    is_dist, local_rank, rank, world_size = setup_distributed()
    is_master = (rank == 0)

    seed_all(1337 + rank)
    assert DEVICE.type == "cuda", "A800 training requires CUDA"
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", local_rank) if is_dist else DEVICE

    cfg = make_config(args.preset, vocab_size=VOCAB_SIZE, block_size=args.block_size)
    model = GPT(cfg)
    raw_params = get_num_params(model)
    if is_master:
        print(f"model params: {raw_params/1e9:.2f}B  preset={args.preset}  dtype={COMPUTE_DTYPE}")
        print(f"distributed: {is_dist}  world_size={world_size}")

    # Warm-start from a prior checkpoint (weights only; optimizer restarts).
    # Must happen BEFORE FSDP wrap so the full state_dict loads cleanly.
    if args.resume and os.path.exists(args.resume):
        state = load_ckpt(args.resume, map_location="cpu")
        try:
            model.load_state_dict(state["model"])
            if is_master:
                print(f"resumed weights from {args.resume} @ step {state.get('step', '?')}")
        except Exception as e:
            if is_master:
                print(f"  [warn] resume failed ({e}); starting from scratch")

    model = model.to(device).to(COMPUTE_DTYPE)
    if is_dist:
        model = wrap_fsdp(model)

    loader = PretrainLoader(
        args.data_dir, args.device_batch_size, args.block_size, device,
        rank=rank, world_size=world_size,
    )
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
            # Avoid cross-rank gradient sync on non-final micro steps (FSDP / DDP)
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

        dt = time.time() - t0
        tok = step * args.grad_accum * args.device_batch_size * args.block_size * world_size
        ktok_s = tok / dt / 1e3
        if is_master:
            writer.add_scalar("train/loss", loss_accum, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/grad_norm", float(grad_norm), step)
            writer.add_scalar("perf/ktok_per_s", ktok_s, step)
            writer.add_scalar("perf/tokens_seen", tok, step)

            if step % args.log_every == 0:
                print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | {ktok_s:.1f} Ktok/s")
        if step % args.save_every == 0 or step == args.max_steps:
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
