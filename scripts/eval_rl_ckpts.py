"""Evaluate RL checkpoints on MBPP test split with Pass@k metrics.

Scans `checkpoints/${RUN}/step_*.pt` (or a single `--ckpt`) and reports
Pass@1 / Pass@k for each, so you can find the best step the same way the
MathGPT A800 report identifies its global-optimum step.

Usage:
    # scan an entire RL run dir
    python -m scripts.eval_rl_ckpts \
        --run-dir checkpoints/codechat_2b_rl \
        --n-samples 16 --ks 1,4,8,16 --limit 100

    # or evaluate a single checkpoint
    python -m scripts.eval_rl_ckpts \
        --ckpt checkpoints/codechat_2b_rl/step_000120.pt
"""
import argparse
import glob
import math
import os
import time

import torch
import torch.nn.functional as F

from codechat.common import DEVICE, COMPUTE_DTYPE, seed_all
from codechat.gpt import GPT, GPTConfig
from codechat.checkpoint import load as load_ckpt
from codechat.tokenizer import encode, decode, USER_TAG, ASSISTANT_TAG, END_TAG
from codechat.execution import run_with_tests, extract_code


def build_prompt(problem: str) -> str:
    user = (
        "Solve the following Python problem. Return ONLY a Python code block.\n\n"
        + problem
    )
    return f"{USER_TAG}\n{user}\n{END_TAG}\n{ASSISTANT_TAG}\n"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k from the Codex paper (n samples, c correct)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def sample_one(model, prompt_ids, max_new, temperature, top_k):
    """Single-sequence sampler matching scripts.chat_rl.sample_one."""
    model.eval()
    ids = prompt_ids.clone()
    new_ids = []
    eot_str_ids = set(encode(END_TAG))
    with torch.no_grad():
        for _ in range(max_new):
            cond = ids[:, -model.cfg.block_size:]
            logits, _ = model(cond)
            logits = logits[:, -1, :].float() / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            new_ids.append(nxt.item())
            ids = torch.cat([ids, nxt], dim=1)
            if new_ids[-1] in eot_str_ids and len(new_ids) > 4:
                break
    return new_ids


def evaluate_ckpt(ckpt_path, problems, n_samples, ks, max_new_tokens, temperature, top_k):
    state = load_ckpt(ckpt_path)
    cfg = GPTConfig(**state["cfg"])
    model = GPT(cfg).to(DEVICE).to(COMPUTE_DTYPE)
    model.load_state_dict(state["model"])
    model.eval()

    per_k = {k: [] for k in ks}
    correct_total = 0
    n_problems = 0
    for ex in problems:
        prompt = build_prompt(ex["prompt"])
        prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
        if prompt_ids.shape[1] > cfg.block_size - max_new_tokens:
            for k in ks:
                per_k[k].append(0.0)
            n_problems += 1
            continue
        c = 0
        for _ in range(n_samples):
            new_ids = sample_one(model, prompt_ids, max_new_tokens, temperature, top_k)
            text = decode(new_ids)
            if END_TAG in text:
                text = text.split(END_TAG)[0]
            code = extract_code(text)
            reward = run_with_tests(code, ex["test_list"])
            if reward >= 0.999:
                c += 1
        correct_total += c
        n_problems += 1
        for k in ks:
            per_k[k].append(pass_at_k(n_samples, c, k))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {k: sum(v) / len(v) for k, v in per_k.items()}, correct_total, n_problems


def discover_ckpts(run_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, "step_*.pt"))):
        base = os.path.basename(p)
        try:
            s = int(base[len("step_"):].split(".")[0])
        except Exception:
            continue
        out.append((s, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", help="checkpoints/${RUN}_rl directory; scans step_*.pt")
    ap.add_argument("--ckpt", help="evaluate a single checkpoint instead")
    ap.add_argument("--n-samples", type=int, default=16, help="completions per problem")
    ap.add_argument("--ks", default="1,4,8,16", help="comma-separated k values")
    ap.add_argument("--limit", type=int, default=100, help="MBPP test problems (0 = all)")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if not (args.run_dir or args.ckpt):
        ap.error("must pass --run-dir or --ckpt")
    seed_all(args.seed)
    assert DEVICE.type == "cuda"
    torch.set_float32_matmul_precision("high")

    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    assert max(ks) <= args.n_samples, f"max k={max(ks)} > n_samples={args.n_samples}"

    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
    problems = list(ds)
    print(f"loaded {len(problems)} MBPP test problems | n_samples={args.n_samples} | ks={ks}")

    if args.ckpt:
        base = os.path.basename(args.ckpt)
        step = 0
        if base.startswith("step_"):
            try:
                step = int(base[len("step_"):].split(".")[0])
            except Exception:
                pass
        ckpts = [(step, args.ckpt)]
    else:
        ckpts = discover_ckpts(args.run_dir)
        if not ckpts:
            raise SystemExit(f"no step_*.pt found in {args.run_dir}")
    print(f"evaluating {len(ckpts)} checkpoint(s)\n")

    header = f"{'step':>7s} | " + " | ".join(f"Pass@{k:<3d}" for k in ks) + " | secs"
    print(header)
    print("-" * len(header))
    results = []
    for step, path in ckpts:
        t0 = time.time()
        scores, correct, n_prob = evaluate_ckpt(
            path, problems, args.n_samples, ks,
            args.max_new_tokens, args.temperature, args.top_k,
        )
        dt = time.time() - t0
        results.append((step, path, scores))
        row = f"{step:>7d} | " + " | ".join(f"{scores[k]*100:6.2f}%" for k in ks)
        print(f"{row} | {dt:5.0f}")

    if len(results) > 1:
        best_k = ks[0]
        best = max(results, key=lambda r: r[2][best_k])
        print(f"\nbest by Pass@{best_k}: step {best[0]} -> {best[2][best_k]*100:.2f}%")
        print(f"  ({best[1]})")


if __name__ == "__main__":
    main()
