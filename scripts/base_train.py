"""Single-GPU pretraining for CodeChat on an A800 80GB, bf16."""
import argparse
import os
import time
import torch

from codechat.common import DEVICE, COMPUTE_DTYPE, seed_all, get_num_params
from codechat.gpt import GPT, GPTConfig
from codechat.dataloader import PretrainLoader
from codechat.optim import build_optimizer, cosine_lr
from codechat.checkpoint import save as save_ckpt
from codechat.tokenizer import VOCAB_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/pretrain")
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--block-size", type=int, default=2048)
    ap.add_argument("--device-batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--run", default="codechat_d20")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    args = ap.parse_args()

    seed_all(1337)
    assert DEVICE.type == "cuda", "A800 training requires CUDA"
    torch.set_float32_matmul_precision("high")

    cfg = GPTConfig(vocab_size=VOCAB_SIZE, depth=args.depth, block_size=args.block_size)
    model = GPT(cfg).to(DEVICE)
    # keep 2D weights in bf16 for compute; RMSNorm/embeddings already follow COMPUTE_DTYPE
    model = model.to(COMPUTE_DTYPE)
    print(f"model params: {get_num_params(model)/1e6:.1f}M  dtype={COMPUTE_DTYPE}")

    loader = PretrainLoader(args.data_dir, args.device_batch_size, args.block_size, DEVICE)
    optim = build_optimizer(model, lr=args.lr)

    ckpt_path = os.path.join(args.ckpt_dir, args.run, "latest.pt")
    t0 = time.time()
    model.train()
    for step in range(1, args.max_steps + 1):
        lr = cosine_lr(step, args.max_steps, args.lr, warmup=args.warmup)
        for g in optim.param_groups:
            g["lr"] = lr

        optim.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            x, y = loader.next_batch()
            _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            loss_accum += loss.item() / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % args.log_every == 0:
            dt = time.time() - t0
            tok = step * args.grad_accum * args.device_batch_size * args.block_size
            print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | {tok/dt/1e3:.1f} Ktok/s")
        if step % args.save_every == 0 or step == args.max_steps:
            save_ckpt(ckpt_path, model, optim, step, cfg)
            print(f"  saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
