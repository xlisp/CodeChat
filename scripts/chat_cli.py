"""CLI chat with a trained CodeChat model."""
import argparse
import torch

from codechat.common import DEVICE, COMPUTE_DTYPE
from codechat.gpt import GPT, GPTConfig
from codechat.checkpoint import load as load_ckpt
from codechat.tokenizer import encode, decode, USER_TAG, ASSISTANT_TAG, END_TAG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    state = load_ckpt(args.ckpt)
    cfg = GPTConfig(**state["cfg"])
    model = GPT(cfg).to(DEVICE).to(COMPUTE_DTYPE)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {args.ckpt}, depth={cfg.depth}, block_size={cfg.block_size}")
    print("type your Python question; Ctrl-C to exit.\n")

    history = ""
    while True:
        try:
            user = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        history += f"{USER_TAG}\n{user}\n{END_TAG}\n{ASSISTANT_TAG}\n"
        ids = torch.tensor([encode(history)], dtype=torch.long, device=DEVICE)
        out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k)
        new_ids = out[0, ids.shape[1]:].tolist()
        text = decode(new_ids)
        # stop at END_TAG if present
        if END_TAG in text:
            text = text.split(END_TAG)[0]
        print(text.strip(), "\n")
        history += text.strip() + f"\n{END_TAG}\n"


if __name__ == "__main__":
    main()
