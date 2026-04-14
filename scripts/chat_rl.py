"""GRPO-style RL on MBPP with executable rewards.

For each prompt we sample G completions from the current policy, run the
reference test asserts in a subprocess, and use (reward - group_mean) /
(group_std + eps) as the advantage. Loss is the standard PG loss with a
KL penalty toward a frozen reference model to keep the policy near the SFT
checkpoint.

Supports both single-GPU and multi-GPU (FSDP) training.  Launch mode is
auto-detected from torchrun env vars, identical to chat_sft.py.

  Single GPU:  python -m scripts.chat_rl --sft-ckpt ...
  8x A800:     torchrun --nproc_per_node=8 -m scripts.chat_rl --sft-ckpt ...
"""
import argparse
import functools
import glob
import json
import os
import time
import copy
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from codechat.common import COMPUTE_DTYPE, seed_all
from codechat.gpt import GPT, GPTConfig, Block
from codechat.optim import build_optimizer, cosine_lr
from codechat.checkpoint import save as save_ckpt, load as load_ckpt
from codechat.tokenizer import encode, decode, USER_TAG, ASSISTANT_TAG, END_TAG
from codechat.execution import run_with_tests, extract_code


def load_problems(problems_file: str | None):
    """Return a list of {'prompt': str, 'test_list': list[str]}.

    If problems_file is given, read it as jsonl (produced by
    scripts/filter_mbpp_by_passrate.py). Otherwise fall back to the full
    MBPP sanitized train split.
    """
    if problems_file and os.path.exists(problems_file):
        items = []
        with open(problems_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    from datasets import load_dataset
    mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    return [{"prompt": ex["prompt"], "test_list": ex["test_list"]} for ex in mbpp]


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


def build_prompt(problem: str) -> str:
    user = (
        "Solve the following Python problem. Return ONLY a Python code block.\n\n"
        + problem
    )
    return f"{USER_TAG}\n{user}\n{END_TAG}\n{ASSISTANT_TAG}\n"


def sample_one(model: GPT, prompt_ids: torch.Tensor, max_new: int,
               temperature: float, top_k: int, block_size: int, is_dist: bool = False):
    """Sample a single completion. Returns (token_ids [T_new], logprobs [T_new]).

    When running under FSDP (is_dist=True), all ranks must call this together
    (FSDP forward requires collective participation).  The sampled token is
    broadcast from rank 0 so every rank stays in sync.
    """
    model.eval()
    ids = prompt_ids.clone()
    new_ids, new_logps = [], []
    eot_str_ids = set(encode(END_TAG))
    with torch.no_grad():
        for _ in range(max_new):
            cond = ids[:, -block_size:]
            logits, _ = model(cond)
            logits = logits[:, -1, :].float() / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            logp_full = F.log_softmax(logits, dim=-1)
            probs = logp_full.exp()
            nxt = torch.multinomial(probs, 1)
            # Ensure all FSDP ranks sample the same token
            if is_dist:
                dist.broadcast(nxt, src=0)
            lp = logp_full.gather(-1, nxt)
            new_ids.append(nxt.item())
            new_logps.append(lp.item())
            ids = torch.cat([ids, nxt], dim=1)
            # crude stop: end tag id present in last few tokens
            if new_ids[-1] in eot_str_ids and len(new_ids) > 4:
                break
    return new_ids, new_logps, ids


def forward_logps(model: GPT, full_ids: torch.Tensor, prompt_len: int):
    """Return per-token logprobs of the completion tokens under `model`."""
    logits, _ = model(full_ids)
    logits = logits[:, :-1, :]  # predict next
    targets = full_ids[:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1)
    tgt_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # only the completion part
    return tgt_logp[:, prompt_len - 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-ckpt", required=True)
    # see scripts/base_train.py for why --run-name is the primary form
    ap.add_argument("--run-name", "--run", dest="run", default="codechat_d20_rl")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--group-size", type=int, default=8,
                    help="completions per prompt. 8 gives group-variance even "
                         "at ~2-3%% per-problem pass rate; 4 is too small.")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-coef", type=float, default=0.02)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--tb-dir", default="runs/tb", help="tensorboard log root")
    ap.add_argument("--reward-mode", choices=("binary", "fractional", "tiered"),
                    default="tiered",
                    help="binary: 1.0 iff all tests pass, else 0.0. "
                         "fractional: passed / total. "
                         "tiered (default): staircase with partial credit for "
                         "parseable / runnable code (see codechat/execution.py).")
    ap.add_argument("--problems-file", default=None,
                    help="jsonl of {prompt,test_list} (from filter_mbpp_by_passrate). "
                         "If unset, use full MBPP sanitized train split.")
    ap.add_argument("--log-rollouts-every", type=int, default=50,
                    help="every N steps, print one decoded rollout + its reward "
                         "(master rank only). Set 0 to disable.")
    ap.add_argument(
        "--save-every", type=int, default=50,
        help="save step_{step:06d}.pt every N steps (in addition to latest.pt)",
    )
    ap.add_argument(
        "--keep-every", type=int, default=50,
        help="among saved step_*.pt files, persistently keep only multiples of this; "
             "non-multiples get deleted at the next save. Set equal to --save-every "
             "to keep every snapshot.",
    )
    args = ap.parse_args()

    is_dist, local_rank, rank, world_size = setup_distributed()
    is_master = (rank == 0)

    seed_all(1337)
    device = torch.device("cuda", local_rank) if is_dist else torch.device("cuda")
    assert device.type == "cuda"
    torch.set_float32_matmul_precision("high")

    # Load checkpoint to CPU so all ranks don't compete for GPU 0 memory.
    state = load_ckpt(args.sft_ckpt, map_location="cpu")
    cfg = GPTConfig(**state["cfg"])

    policy = GPT(cfg)
    policy.load_state_dict(state["model"])
    ref = GPT(cfg)
    ref.load_state_dict(state["model"])
    del state  # free CPU memory

    policy = policy.to(device).to(COMPUTE_DTYPE)
    ref = ref.to(device).to(COMPUTE_DTYPE)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()

    if is_dist:
        policy = wrap_fsdp(policy)
        ref = wrap_fsdp(ref)

    if is_master:
        print(f"loaded sft ckpt {args.sft_ckpt}")

    problems = load_problems(args.problems_file)
    if is_master:
        src = args.problems_file or "google-research-datasets/mbpp:train"
        print(f"loaded {len(problems)} problems from {src}")
        print(f"reward_mode={args.reward_mode}  group_size={args.group_size}")

    optim = build_optimizer(policy, lr=args.lr)
    ckpt_path = os.path.join(args.ckpt_dir, args.run, "latest.pt")
    tb_path = os.path.join(args.tb_dir, args.run)
    writer = SummaryWriter(log_dir=tb_path) if is_master else None
    if is_master:
        print(f"tensorboard -> {tb_path}")
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        lr = cosine_lr(step, args.max_steps, args.lr, warmup=20)
        for g in optim.param_groups:
            g["lr"] = lr

        # All ranks must pick the same problem — broadcast index from rank 0.
        idx = torch.randint(0, len(problems), (1,), device=device)
        if is_dist:
            dist.broadcast(idx, src=0)
        ex = problems[idx.item()]
        problem = ex["prompt"]
        tests = ex["test_list"]
        prompt = build_prompt(problem)
        prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        prompt_len = prompt_ids.shape[1]
        if prompt_len > cfg.block_size - args.max_new_tokens:
            continue

        # ---- sample a group of completions and score ----
        rollouts = []  # list of (full_ids, reward, decoded_text)
        for _ in range(args.group_size):
            new_ids, _old_lp, full_ids = sample_one(
                policy, prompt_ids, args.max_new_tokens, args.temperature, args.top_k,
                block_size=cfg.block_size, is_dist=is_dist,
            )
            text = decode(new_ids)
            if END_TAG in text:
                text = text.split(END_TAG)[0]
            code = extract_code(text)
            if args.reward_mode == "binary":
                r_frac = run_with_tests(code, tests, mode="fractional")
                reward = 1.0 if r_frac >= 0.999 else 0.0
            else:
                reward = run_with_tests(code, tests, mode=args.reward_mode)
            rollouts.append((full_ids, reward, text))

        rewards = torch.tensor([r for _, r, _ in rollouts], dtype=torch.float32, device=device)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        # ---- PG + KL update ----
        policy.train()
        optim.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_pg = 0.0
        total_kl = 0.0
        for (full_ids, _, _), a in zip(rollouts, adv):
            logp_pol = forward_logps(policy, full_ids, prompt_len)
            with torch.no_grad():
                logp_ref = forward_logps(ref, full_ids, prompt_len)
            # per-token KL approximation (k1): logp_pol - logp_ref
            kl = (logp_pol - logp_ref).mean()
            # PG loss: -advantage * sum logp
            pg = -a * logp_pol.mean()
            loss = pg + args.kl_coef * kl
            (loss / args.group_size).backward()
            total_loss += loss.item() / args.group_size
            total_pg += pg.item() / args.group_size
            total_kl += kl.item() / args.group_size
        if is_dist and hasattr(policy, "clip_grad_norm_"):
            grad_norm = policy.clip_grad_norm_(args.clip * 5)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip * 5)
        optim.step()

        if is_master:
            writer.add_scalar("rl/reward_mean", rewards.mean().item(), step)
            writer.add_scalar("rl/reward_max", rewards.max().item(), step)
            writer.add_scalar("rl/reward_min", rewards.min().item(), step)
            writer.add_scalar("rl/reward_std", rewards.std().item(), step)
            writer.add_scalar("rl/advantage_abs_mean", adv.abs().mean().item(), step)
            writer.add_scalar("rl/loss_total", total_loss, step)
            writer.add_scalar("rl/loss_pg", total_pg, step)
            writer.add_scalar("rl/kl", total_kl, step)
            writer.add_scalar("rl/lr", lr, step)
            writer.add_scalar("rl/grad_norm", float(grad_norm), step)
            writer.add_scalar("rl/prompt_len", prompt_len, step)
            writer.add_scalar("rl/elapsed_s", time.time() - t0, step)

        if step % 5 == 0 and is_master:
            print(
                f"rl step {step:5d} | reward {rewards.mean().item():.3f} "
                f"(max {rewards.max().item():.2f}) | loss {total_loss:.4f} "
                f"| lr {lr:.2e} | {time.time()-t0:.0f}s"
            )

        # Periodically dump one rollout so we can eyeball what the model is
        # actually generating — essential when reward stays flat.
        if (args.log_rollouts_every and is_master
                and step % args.log_rollouts_every == 0):
            # Pick the best rollout in this group so we see near-successes, not random noise.
            best_i = int(torch.argmax(rewards).item())
            best_r = rewards[best_i].item()
            best_text = rollouts[best_i][2]
            snippet = best_text[:800].replace("\n", "\n    ")
            print(
                f"  [rollout @ step {step}] best_reward={best_r:.3f}  "
                f"(rewards={[round(r, 3) for r in rewards.tolist()]})\n"
                f"    prompt: {problem[:200]!r}\n"
                f"    tests:  {tests[:1]}\n"
                f"    best output >>>\n    {snippet}\n    <<<"
            )
            if writer is not None:
                writer.add_text(
                    f"rl/rollout_best",
                    f"step {step} | reward {best_r:.3f}\n\n"
                    f"prompt:\n{problem}\n\ntests:\n{tests}\n\noutput:\n{best_text[:2000]}",
                    step,
                )
        if step % args.save_every == 0 or step == args.max_steps:
            save_ckpt(ckpt_path, policy, optim, step, cfg)
            step_path = os.path.join(args.ckpt_dir, args.run, f"step_{step:06d}.pt")
            save_ckpt(step_path, policy, optim, step, cfg)
            # rotate older snapshots: keep only multiples of --keep-every
            if args.keep_every > args.save_every:
                for p in glob.glob(os.path.join(args.ckpt_dir, args.run, "step_*.pt")):
                    base = os.path.basename(p)
                    try:
                        s = int(base[len("step_"):].split(".")[0])
                    except Exception:
                        continue
                    if s == step:
                        continue
                    if s % args.keep_every != 0:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
            if is_master:
                print(f"  saved -> {ckpt_path} and {step_path}")
    if is_master:
        writer.close()
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
