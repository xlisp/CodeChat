"""Supervised fine-tuning on Python instruction data."""
import argparse
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from codechat.common import DEVICE, COMPUTE_DTYPE, seed_all
from codechat.gpt import GPT, GPTConfig
from codechat.dataloader import SFTLoader
from codechat.optim import build_optimizer, cosine_lr
from codechat.checkpoint import save as save_ckpt, load as load_ckpt


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

    seed_all(1337)
    assert DEVICE.type == "cuda"
    torch.set_float32_matmul_precision("high")

    state = load_ckpt(args.base_ckpt)
    cfg = GPTConfig(**state["cfg"])
    model = GPT(cfg).to(DEVICE).to(COMPUTE_DTYPE)
    model.load_state_dict(state["model"])
    print(f"loaded base ckpt {args.base_ckpt} @ step {state['step']}")

    loader = SFTLoader(os.path.join(args.data_dir, "train.jsonl"),
                       args.device_batch_size, cfg.block_size, DEVICE)
    optim = build_optimizer(model, lr=args.lr)

    ckpt_path = os.path.join(args.ckpt_dir, args.run, "latest.pt")
    tb_path = os.path.join(args.tb_dir, args.run)
    writer = SummaryWriter(log_dir=tb_path)
    print(f"tensorboard -> {tb_path}")
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
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        writer.add_scalar("sft/loss", loss_accum, step)
        writer.add_scalar("sft/lr", lr, step)
        writer.add_scalar("sft/grad_norm", float(grad_norm), step)
        writer.add_scalar("sft/elapsed_s", time.time() - t0, step)

        if step % 20 == 0:
            print(f"sft step {step:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | {time.time()-t0:.0f}s")
        if step % 500 == 0 or step == args.max_steps:
            save_ckpt(ckpt_path, model, optim, step, cfg)
            print(f"  saved -> {ckpt_path}")
    writer.close()


if __name__ == "__main__":
    main()
