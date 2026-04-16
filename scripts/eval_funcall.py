"""Evaluate a checkpoint (or a whole run dir) on the funcall eval set.

Reports Pass@k (unbiased Codex formula), full-match rate, and the full
reward-tier distribution from codechat.funcall_reward. Also prints a
handful of sample rollouts so you can eyeball failure modes (e.g. the
JSON-in-JSON quote-escape issue we saw during RL step 50).

Usage:
    # single ckpt — the common case
    python -m scripts.eval_funcall \
        --ckpt checkpoints/codechat_8b_rl_funcall_v5/step_000060.pt \
        --num-samples 16 --ks 1,4,8,16

    # compare SFT vs RL quickly
    python -m scripts.eval_funcall \
        --run-dir checkpoints/codechat_8b_rl_funcall_v5 --num-samples 16

    # or pass --ckpt multiple times to compare an arbitrary set
    python -m scripts.eval_funcall \
        --ckpt checkpoints/codechat_8b_funcall_v5/latest.pt \
        --ckpt checkpoints/codechat_8b_rl_funcall_v5/step_000060.pt
"""
import argparse
import glob
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from codechat.common import DEVICE, COMPUTE_DTYPE, seed_all
from codechat.gpt import GPT, GPTConfig
from codechat.checkpoint import load as load_ckpt
from codechat.tokenizer import encode, decode, END_TAG
from codechat.funcall_reward import funcall_reward


TIER_ORDER = ["no_tag", "bad_json", "no_name", "wrong_name",
              "name_only", "partial", "full_match"]


def load_problems(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k (Codex paper). n samples, c exact-match correct."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


@torch.no_grad()
def sample_batch(model, prompt_ids, num_samples, max_new_tokens,
                 temperature, top_k, block_size):
    """Generate `num_samples` completions from one prompt in one batch."""
    model.eval()
    ids = prompt_ids.repeat(num_samples, 1)  # [K, T]
    eot_ids = set(encode(END_TAG))
    done = torch.zeros(num_samples, dtype=torch.bool, device=ids.device)
    new_ids = [[] for _ in range(num_samples)]

    for _ in range(max_new_tokens):
        cond = ids[:, -block_size:]
        logits, _ = model(cond)
        step_logits = logits[:, -1, :].float() / max(temperature, 1e-5)
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(step_logits, min(top_k, step_logits.size(-1)))
            step_logits[step_logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(step_logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
        nxt_cpu = nxt.squeeze(-1).tolist()
        for i in range(num_samples):
            if done[i]:
                continue
            tok = nxt_cpu[i]
            new_ids[i].append(tok)
            if tok in eot_ids and len(new_ids[i]) > 4:
                done[i] = True
        if bool(done.all()):
            break
    return new_ids


def _tier_bucket(tier: str) -> str:
    """Collapse partial_0.XX buckets into a single 'partial' row."""
    return "partial" if tier.startswith("partial_") else tier


def evaluate_ckpt(ckpt_path, problems, num_samples, ks, max_new_tokens,
                  temperature, top_k, show_examples):
    state = load_ckpt(ckpt_path, map_location="cpu")
    cfg = GPTConfig(**state["cfg"])
    model = GPT(cfg).to(DEVICE).to(COMPUTE_DTYPE)
    model.load_state_dict(state["model"])
    del state
    model.eval()

    max_prompt_len = cfg.block_size - max_new_tokens
    per_k = {k: [] for k in ks}
    tier_counts: dict[str, int] = {}
    full_match_count = 0
    sampled_count = 0
    reward_sum = 0.0
    examples_shown = 0

    for ex_i, ex in enumerate(problems):
        prompt_ids = torch.tensor([encode(ex["prompt"])], dtype=torch.long, device=DEVICE)
        if prompt_ids.shape[1] > max_prompt_len:
            # Same left-truncation policy as training.
            prompt_ids = prompt_ids[:, -max_prompt_len:]

        new_ids_list = sample_batch(
            model, prompt_ids, num_samples, max_new_tokens,
            temperature, top_k, cfg.block_size,
        )

        c = 0
        rewards_this = []
        tiers_this = []
        texts_this = []
        for ids in new_ids_list:
            text = decode(ids)
            if END_TAG in text:
                text = text.split(END_TAG)[0]
            r, tier = funcall_reward(text, ex["gt_name"], ex["gt_args"])
            rewards_this.append(r)
            tiers_this.append(tier)
            texts_this.append(text)
            bucket = _tier_bucket(tier)
            tier_counts[bucket] = tier_counts.get(bucket, 0) + 1
            sampled_count += 1
            reward_sum += r
            if r >= 0.999:
                c += 1
        full_match_count += c
        for k in ks:
            per_k[k].append(pass_at_k(num_samples, c, k))

        if examples_shown < show_examples:
            best_i = max(range(num_samples), key=lambda i: rewards_this[i])
            snippet = texts_this[best_i].strip()
            if len(snippet) > 600:
                snippet = snippet[:600] + " ...[truncated]"
            snippet = snippet.replace("\n", "\n  ")
            print(f"\n  --- example {ex_i + 1} ---")
            print(f"  gt_name: {ex['gt_name']}")
            print(f"  gt_args: {ex['gt_args']}")
            print(f"  rewards: {[round(r, 2) for r in rewards_this]}  "
                  f"(exact={c}/{num_samples})")
            print(f"  best tier={tiers_this[best_i]}  reward={rewards_this[best_i]:.2f}")
            print(f"  best output:\n  {snippet}")
            examples_shown += 1

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "passk": {k: sum(v) / len(v) for k, v in per_k.items()},
        "tier_counts": tier_counts,
        "full_match": full_match_count,
        "sampled": sampled_count,
        "avg_reward": reward_sum / max(1, sampled_count),
        "n_problems": len(problems),
    }


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


def _step_from_path(path):
    base = os.path.basename(path)
    if base.startswith("step_"):
        try:
            return int(base[len("step_"):].split(".")[0])
        except Exception:
            pass
    return 0


def print_result(step, path, res, ks, dt):
    print(f"\n[step {step}]  {path}")
    for k in ks:
        print(f"  Pass@{k:<3d} = {res['passk'][k] * 100:6.2f}%")
    if res["sampled"]:
        print(f"  full-match     = {res['full_match']}/{res['sampled']} "
              f"({res['full_match'] / res['sampled'] * 100:.2f}%)")
        print(f"  avg reward     = {res['avg_reward']:.3f}")
        print(f"  tier distribution (of {res['sampled']} rollouts):")
        for t in TIER_ORDER:
            if t in res["tier_counts"]:
                c = res["tier_counts"][t]
                print(f"    {t:<12s} {c:5d}  ({c / res['sampled'] * 100:5.2f}%)")
    print(f"  elapsed: {dt:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=None,
                    help="path to a checkpoint; may be passed multiple times")
    ap.add_argument("--run-dir",
                    help="scan step_*.pt inside this run dir")
    ap.add_argument("--eval-file", default="data/rl_funcall/eval.jsonl")
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--ks", default="1,4,8,16")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all eval problems")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--show-examples", type=int, default=5,
                    help="print first N rollouts (only for single-ckpt mode)")
    args = ap.parse_args()

    if not (args.ckpt or args.run_dir):
        ap.error("pass --ckpt <path> (repeatable) or --run-dir <dir>")
    seed_all(args.seed)
    assert DEVICE.type == "cuda", "eval requires a GPU"
    torch.set_float32_matmul_precision("high")

    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    assert max(ks) <= args.num_samples, (
        f"max k={max(ks)} > num_samples={args.num_samples}"
    )

    problems = load_problems(args.eval_file)
    if args.limit > 0:
        problems = problems[:args.limit]
    print(f"loaded {len(problems)} funcall eval problems from {args.eval_file}")
    print(f"num_samples={args.num_samples}  ks={ks}  "
          f"temp={args.temperature}  top_k={args.top_k}  "
          f"max_new_tokens={args.max_new_tokens}")

    ckpts: list[tuple[int, str]] = []
    if args.ckpt:
        for p in args.ckpt:
            ckpts.append((_step_from_path(p), p))
    if args.run_dir:
        ckpts.extend(discover_ckpts(args.run_dir))
    if not ckpts:
        raise SystemExit("no checkpoints found")
    print(f"evaluating {len(ckpts)} checkpoint(s)")

    all_results = []
    for step, path in ckpts:
        print(f"\n==> {path}")
        t0 = time.time()
        res = evaluate_ckpt(
            path, problems, args.num_samples, ks,
            args.max_new_tokens, args.temperature, args.top_k,
            args.show_examples if len(ckpts) == 1 else 0,
        )
        dt = time.time() - t0
        all_results.append((step, path, res, dt))
        print_result(step, path, res, ks, dt)

    if len(all_results) > 1:
        header = f"\n{'step':>7s} | " + " | ".join(f"Pass@{k:<3d}" for k in ks) \
            + f" | {'full%':>6s} | secs"
        print(header)
        print("-" * len(header))
        for step, path, res, dt in all_results:
            row = f"{step:>7d} | " + " | ".join(
                f"{res['passk'][k] * 100:6.2f}%" for k in ks
            )
            full_pct = res["full_match"] / max(1, res["sampled"]) * 100
            print(f"{row} | {full_pct:5.2f}% | {dt:5.0f}")
        best_k = max(ks)
        best = max(all_results, key=lambda r: r[2]["passk"][best_k])
        print(f"\nbest by Pass@{best_k}: step {best[0]} "
              f"-> {best[2]['passk'][best_k] * 100:.2f}%")
        print(f"  ({best[1]})")


if __name__ == "__main__":
    main()
