"""Filter MBPP problems by the SFT model's pass rate to produce an RL-friendly
problem set.

The trick behind MBPP RL on a weak base: GRPO only has gradient signal when
group rewards have variance. If every problem is either "always fails" or
"always passes", advantage is 0 on every rollout. Keeping only problems
whose pass rate lies in (low, high) — e.g. (0.1, 0.9) — maximizes the
fraction of steps that contribute to learning.

Input: a per-problem breakdown jsonl from eval_mbpp_pass_at_k.py
       (each row has {prompt, test_list, pass_rate, n_pass, k, mean_tiered}).

Output: a jsonl consumable by chat_rl.py via --problems-file — contains
        only {prompt, test_list} rows that pass the pass-rate filter.

Usage:
    # 1) first run the diagnostic, asking it to dump per-problem stats:
    torchrun --nproc_per_node=8 -m scripts.eval_mbpp_pass_at_k \
        --ckpt checkpoints/codechat_8b_sft/latest.pt \
        --n-problems 374 --k 8 \
        --out-jsonl data/mbpp_passrate.jsonl

    # 2) filter to keep problems with pass rate in (0.05, 0.90):
    python -m scripts.filter_mbpp_by_passrate \
        --in-jsonl data/mbpp_passrate.jsonl \
        --out-jsonl data/mbpp_rl_curriculum.jsonl \
        --min-pass-rate 0.05 --max-pass-rate 0.90

    # 3) feed to chat_rl:
    torchrun --nproc_per_node=8 -m scripts.chat_rl \
        --sft-ckpt checkpoints/codechat_8b_sft/latest.pt \
        --problems-file data/mbpp_rl_curriculum.jsonl \
        --reward-mode tiered --group-size 8 ...

If the diagnostic finds no problem with pass_rate > min-pass-rate, the script
will emit a warning and (by default) fall back to keeping problems whose
mean_tiered reward is nonzero — gives the tiered reward at least something
to work with even before any test passes.
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True,
                    help="per-problem breakdown from eval_mbpp_pass_at_k --out-jsonl")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--min-pass-rate", type=float, default=0.05)
    ap.add_argument("--max-pass-rate", type=float, default=0.95)
    ap.add_argument("--min-tiered", type=float, default=0.10,
                    help="fallback threshold on mean_tiered if the pass-rate "
                         "window keeps nothing (very weak base case).")
    args = ap.parse_args()

    with open(args.in_jsonl) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if not rows:
        raise SystemExit(f"no rows in {args.in_jsonl}")

    kept = [r for r in rows
            if args.min_pass_rate <= r.get("pass_rate", 0.0) <= args.max_pass_rate]
    if not kept:
        print(f"WARN: no problems fell in pass_rate in "
              f"[{args.min_pass_rate}, {args.max_pass_rate}]; "
              f"falling back to mean_tiered >= {args.min_tiered}")
        kept = [r for r in rows if r.get("mean_tiered", 0.0) >= args.min_tiered]

    if not kept:
        raise SystemExit(
            "still nothing after fallback — base model can't even produce "
            "parseable code. Improve the SFT stage before running RL."
        )

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in kept:
            out = {"prompt": r["prompt"], "test_list": r["test_list"]}
            f.write(json.dumps(out) + "\n")

    # summary
    pr = [r.get("pass_rate", 0.0) for r in kept]
    mt = [r.get("mean_tiered", 0.0) for r in kept]
    print(f"input            : {len(rows)} problems")
    print(f"output           : {len(kept)} problems -> {args.out_jsonl}")
    print(f"kept pass_rate   : mean={sum(pr)/len(pr):.3f}  "
          f"min={min(pr):.3f}  max={max(pr):.3f}")
    print(f"kept mean_tiered : mean={sum(mt)/len(mt):.3f}  "
          f"min={min(mt):.3f}  max={max(mt):.3f}")


if __name__ == "__main__":
    main()
