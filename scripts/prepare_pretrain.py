"""Download the Python subset of codeparrot/github-code-clean, tokenize with
GPT-2 BPE and write flat uint16 .bin shards to --out-dir.

Usage:
    python -m scripts.prepare_pretrain --out-dir data/pretrain --max-shards 8
"""
import argparse
import os
import numpy as np
from tqdm import tqdm

from codechat.tokenizer import encode, EOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/pretrain")
    ap.add_argument("--max-shards", type=int, default=8, help="number of 256MB shards to write")
    ap.add_argument("--shard-tokens", type=int, default=128 * 1024 * 1024, help="tokens per shard (~256MB at uint16)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset(
        "codeparrot/github-code-clean",
        "Python-all",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    buf = np.empty(args.shard_tokens, dtype=np.uint16)
    pos = 0
    shard_idx = 0
    pbar = tqdm(desc=f"shard {shard_idx}", total=args.shard_tokens)
    for ex in ds:
        code = ex.get("code") or ""
        if not code:
            continue
        ids = encode(code) + [EOT]
        if pos + len(ids) > args.shard_tokens:
            take = args.shard_tokens - pos
            buf[pos:] = ids[:take]
            shard_path = os.path.join(args.out_dir, f"shard_{shard_idx:04d}.bin")
            buf.tofile(shard_path)
            shard_idx += 1
            pbar.close()
            if shard_idx >= args.max_shards:
                return
            # start next shard with leftover
            leftover = ids[take:]
            pos = len(leftover)
            buf[:pos] = leftover
            pbar = tqdm(desc=f"shard {shard_idx}", total=args.shard_tokens, initial=pos)
        else:
            buf[pos : pos + len(ids)] = ids
            pos += len(ids)
            pbar.update(len(ids))


if __name__ == "__main__":
    main()
