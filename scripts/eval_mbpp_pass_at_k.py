"""pass@k diagnostic on MBPP — decides whether RL has a chance.

Run this on the SFT checkpoint BEFORE launching RL. If pass@k ≈ 0% the RL
reward will stay flat (no gradient signal) and you'd just waste GPU hours —
fix the base model first (more SFT, better data, larger group, tiered reward).

Usage:
    # single GPU (works for smaller models):
    python -m scripts.eval_mbpp_pass_at_k \
        --ckpt checkpoints/codechat_8b_sft/latest.pt \
        --n-problems 50 --k 8

    # 8x A800 (required for 8B bf16 — policy weights are too big for one card):
    torchrun --nproc_per_node=8 -m scripts.eval_mbpp_pass_at_k \
        --ckpt checkpoints/codechat_8b_sft/latest.pt \
        --n-problems 50 --k 8

Reports: pass@1, pass@k, tiered-reward mean, runnable-rate, parseable-rate.
Also writes a per-problem breakdown to --out-jsonl so you can feed it to
filter_mbpp_by_passrate.py afterwards.
"""
import argparse
import functools
import json
import os
import time

import torch
import torch.distributed as dist

from codechat.common import COMPUTE_DTYPE, seed_all
from codechat.gpt import GPT, GPTConfig, Block
from codechat.checkpoint import load as load_ckpt
from codechat.tokenizer import encode, decode, USER_TAG, ASSISTANT_TAG, END_TAG
from codechat.execution import run_with_tests, extract_code


def setup_distributed():
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, local_rank, dist.get_rank(), dist.get_world_size()


def wrap_fsdp(model):
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


def build_prompt(problem: str) -> str:
    user = (
        "Solve the following Python problem. Return ONLY a Python code block.\n\n"
        + problem
    )
    return f"{USER_TAG}\n{user}\n{END_TAG}\n{ASSISTANT_TAG}\n"


@torch.no_grad()
def sample_one(model, prompt_ids, max_new, temperature, top_k, block_size, is_dist):
    import torch.nn.functional as F
    model.eval()
    ids = prompt_ids.clone()
    eot_str_ids = set(encode(END_TAG))
    new_ids = []
    for _ in range(max_new):
        cond = ids[:, -block_size:]
        logits, _ = model(cond)
        logits = logits[:, -1, :].float() / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        if is_dist:
            dist.broadcast(nxt, src=0)
        new_ids.append(nxt.item())
        ids = torch.cat([ids, nxt], dim=1)
        if new_ids[-1] in eot_str_ids and len(new_ids) > 4:
            break
    return new_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="train", choices=("train", "test", "validation", "prompt"))
    ap.add_argument("--n-problems", type=int, default=50,
                    help="evaluate on the first N problems of the split")
    ap.add_argument("--k", type=int, default=8, help="samples per problem")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--out-jsonl", default=None,
                    help="if set, write per-problem {prompt,test_list,pass_rate,k} rows")
    args = ap.parse_args()

    is_dist, local_rank, rank, world_size = setup_distributed()
    is_master = (rank == 0)

    seed_all(1337 + rank)
    device = torch.device("cuda", local_rank) if is_dist else torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    state = load_ckpt(args.ckpt, map_location="cpu")
    cfg = GPTConfig(**state["cfg"])
    model = GPT(cfg)
    model.load_state_dict(state["model"])
    del state
    model = model.to(device).to(COMPUTE_DTYPE)
    if is_dist:
        model = wrap_fsdp(model)
    if is_master:
        print(f"loaded {args.ckpt}")

    from datasets import load_dataset
    mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split=args.split)
    n = min(args.n_problems, len(mbpp))
    if is_master:
        print(f"evaluating {n} problems from split={args.split}, k={args.k}")

    per_problem = []
    t0 = time.time()
    total_reward_tiered = 0.0
    total_runnable = 0  # reward >= 0.15
    total_parseable = 0  # reward > 0

    for pi in range(n):
        ex = mbpp[pi]
        problem = ex["prompt"]
        tests = ex["test_list"]
        prompt = build_prompt(problem)
        prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        if prompt_ids.shape[1] > cfg.block_size - args.max_new_tokens:
            continue

        n_pass = 0
        sample_rewards = []
        for _ in range(args.k):
            new_ids = sample_one(model, prompt_ids, args.max_new_tokens,
                                 args.temperature, args.top_k,
                                 block_size=cfg.block_size, is_dist=is_dist)
            text = decode(new_ids)
            if END_TAG in text:
                text = text.split(END_TAG)[0]
            code = extract_code(text)
            r_binary = run_with_tests(code, tests, mode="fractional")
            r_tiered = run_with_tests(code, tests, mode="tiered")
            if r_binary >= 0.999:
                n_pass += 1
            sample_rewards.append((r_binary, r_tiered))
            total_reward_tiered += r_tiered
            if r_tiered >= 0.15:
                total_runnable += 1
            if r_tiered > 0:
                total_parseable += 1

        pass_rate = n_pass / args.k
        per_problem.append({
            "prompt": problem,
            "test_list": tests,
            "pass_rate": pass_rate,
            "n_pass": n_pass,
            "k": args.k,
            "mean_tiered": sum(t for _, t in sample_rewards) / args.k,
        })
        if is_master and (pi + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  [{pi+1}/{n}] pass_rate={pass_rate:.2f} "
                  f"elapsed={elapsed:.0f}s")

    total_samples = n * args.k
    pass1 = sum(1 for p in per_problem if p["n_pass"] >= 1) / max(len(per_problem), 1)
    # Aggregate pass@1 across all samples (not per-problem):
    raw_pass1 = sum(p["n_pass"] for p in per_problem) / max(total_samples, 1)
    mean_tiered = total_reward_tiered / max(total_samples, 1)
    runnable_rate = total_runnable / max(total_samples, 1)
    parseable_rate = total_parseable / max(total_samples, 1)

    if is_master:
        print("\n" + "=" * 64)
        print(f"MBPP pass@k diagnostic — {args.ckpt}")
        print("=" * 64)
        print(f"problems evaluated : {len(per_problem)}")
        print(f"samples per problem: {args.k}")
        print(f"total samples      : {total_samples}")
        print(f"")
        print(f"pass@1  (avg per sample)   : {raw_pass1:.4f}  ({raw_pass1*100:.2f}%)")
        print(f"pass@{args.k} (any of k correct): {pass1:.4f}  ({pass1*100:.2f}%)")
        print(f"mean tiered reward         : {mean_tiered:.4f}")
        print(f"runnable rate (exec OK)    : {runnable_rate:.4f}  ({runnable_rate*100:.2f}%)")
        print(f"parseable rate (ast OK)    : {parseable_rate:.4f}  ({parseable_rate*100:.2f}%)")
        print("")
        # Interpretation hints
        if raw_pass1 < 0.01:
            print("VERDICT: pass@1 < 1%. GRPO with binary/fractional reward will likely")
            print("         stall. Use --reward-mode tiered AND filter to problems with")
            print("         some signal, OR strengthen the base model (more SFT) first.")
        elif raw_pass1 < 0.05:
            print("VERDICT: pass@1 in 1-5%. RL is possible with tiered reward + group>=8")
            print("         + MBPP difficulty filter. Don't expect miracles.")
        else:
            print("VERDICT: pass@1 >= 5%. Standard GRPO should work. Tiered reward")
            print("         optional but still helps early stability.")
        print("=" * 64)

    if args.out_jsonl and is_master:
        os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
        with open(args.out_jsonl, "w") as f:
            for p in per_problem:
                f.write(json.dumps(p) + "\n")
        print(f"wrote per-problem breakdown -> {args.out_jsonl}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
