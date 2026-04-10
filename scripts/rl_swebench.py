"""GRPO-style RL on SWE-bench train with Docker-based rewards.

Same GRPO algorithm as chat_rl.py (which uses MBPP), but the RL environment
is SWE-bench train: the model reads a GitHub issue and generates a unified-diff
patch, then Docker containers apply the patch and run tests to compute reward.

This is *very* compute-heavy: each reward evaluation spins up a Docker container.
Expect ~1-5 minutes per RL step depending on --group-size and --docker-workers.

Reward modes (from lightest to heaviest):
  "syntax"     — regex check that output looks like a unified diff (no Docker)
  "apply-only" — `git apply --check` inside Docker (~5-10s per eval)
  "docker"     — apply patch + run full test suite (~30-300s per eval)

Recommended ramp-up:
  1. Start with --reward-mode syntax --max-steps 200  to teach diff format
  2. Switch to --reward-mode apply-only --max-steps 300  to teach valid patches
  3. Finish with --reward-mode docker --max-steps 500  for test-pass signal

Single-GPU, bf16, A800 80GB friendly (GPU is mostly idle during Docker eval).

Usage:
    # Phase 1: syntax-only warm-up (fast, no Docker)
    python -m scripts.rl_swebench \\
        --sft-ckpt checkpoints/codechat_2b_sft/latest.pt \\
        --reward-mode syntax --max-steps 200

    # Phase 2: full Docker evaluation
    python -m scripts.rl_swebench \\
        --sft-ckpt checkpoints/codechat_2b_rl_swebench/latest.pt \\
        --reward-mode docker --max-steps 500 --docker-workers 4

    # Pre-build Docker images (one-time, can take hours):
    python -m scripts.rl_swebench \\
        --sft-ckpt checkpoints/codechat_2b_sft/latest.pt \\
        --prepare-images --max-steps 0
"""
import argparse
import glob
import os
import re
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from codechat.common import DEVICE, COMPUTE_DTYPE, seed_all
from codechat.gpt import GPT, GPTConfig
from codechat.optim import build_optimizer, cosine_lr
from codechat.checkpoint import save as save_ckpt, load as load_ckpt
from codechat.tokenizer import encode, decode, USER_TAG, ASSISTANT_TAG, END_TAG
from codechat.swebench_reward import SWEBenchReward


# ---------------------------------------------------------------------------
# Prompt construction (mirrors scripts/eval_swebench.py)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "You are an expert Python developer. You will be given a GitHub issue "
    "describing a bug in an open-source Python repository. Produce a minimal "
    "unified-diff patch that fixes the issue. Respond with ONLY a single "
    "```diff ...``` fenced block. The diff must apply cleanly with "
    "`git apply` from the repository root."
)


def build_prompt(instance: dict) -> str:
    """Build a chat prompt from a SWE-bench instance."""
    problem = instance.get("problem_statement", "") or ""
    repo = instance.get("repo", "")
    base = instance.get("base_commit", "")[:12]
    hints = instance.get("hints_text", "") or ""

    user = f"{SYSTEM_INSTRUCTION}\n\nRepo: {repo}\nBase commit: {base}\n\n"
    user += f"--- Issue ---\n{problem}\n"
    if hints:
        user += f"\n--- Maintainer hints ---\n{hints[:2000]}\n"
    user += "\nReturn the fix as a unified diff now."
    return f"{USER_TAG}\n{user}\n{END_TAG}\n{ASSISTANT_TAG}\n"


def extract_diff(text: str) -> str:
    """Pull a diff out of a model response (same logic as eval_swebench.py).

    Tries ```diff ...``` fenced blocks first, then looks for bare
    ``diff --git`` lines anywhere in the response.
    """
    if "```" in text:
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            chunk = parts[i]
            for lang in ("diff", "patch"):
                if chunk.startswith(lang):
                    chunk = chunk[len(lang):]
                    break
            chunk = chunk.strip()
            if "diff --git" in chunk or chunk.startswith("---"):
                return chunk
    idx = text.find("diff --git")
    if idx >= 0:
        return text[idx:].strip()
    return ""


# ---------------------------------------------------------------------------
# Sampling & log-prob computation (same core as chat_rl.py)
# ---------------------------------------------------------------------------

def sample_one(
    model: GPT,
    prompt_ids: torch.Tensor,
    max_new: int,
    temperature: float,
    top_k: int,
):
    """Sample a single completion. Returns (new_ids, logprobs, full_ids)."""
    model.eval()
    ids = prompt_ids.clone()
    new_ids, new_logps = [], []
    eot_str_ids = set(encode(END_TAG))
    with torch.no_grad():
        for _ in range(max_new):
            cond = ids[:, -model.cfg.block_size :]
            logits, _ = model(cond)
            logits = logits[:, -1, :].float() / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            logp_full = F.log_softmax(logits, dim=-1)
            probs = logp_full.exp()
            nxt = torch.multinomial(probs, 1)
            lp = logp_full.gather(-1, nxt)
            new_ids.append(nxt.item())
            new_logps.append(lp.item())
            ids = torch.cat([ids, nxt], dim=1)
            # stop on end-of-turn token
            if new_ids[-1] in eot_str_ids and len(new_ids) > 4:
                break
    return new_ids, new_logps, ids


def forward_logps(model: GPT, full_ids: torch.Tensor, prompt_len: int):
    """Return per-token log-probs of the completion tokens under *model*."""
    logits, _ = model(full_ids)
    logits = logits[:, :-1, :]  # predict next token
    targets = full_ids[:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1)
    tgt_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return tgt_logp[:, prompt_len - 1 :]


# ---------------------------------------------------------------------------
# Docker image preparation
# ---------------------------------------------------------------------------

def _prepare_images(ds):
    """Build Docker images for all train instances (one-time setup)."""
    try:
        from swebench.harness.docker_build import build_env_images
        from swebench.harness.test_spec import make_test_spec
    except ImportError:
        print(
            "WARNING: swebench Docker build APIs not available.\n"
            "Build images manually:\n"
            "  python -m swebench.harness.run_evaluation "
            "--predictions_path <dummy.jsonl> --build_only\n"
            "Or see https://github.com/princeton-nlp/SWE-bench"
        )
        return

    specs = []
    for i in range(len(ds)):
        try:
            specs.append(make_test_spec(ds[i]))
        except Exception as e:
            print(f"  skip {ds[i]['instance_id']}: {e}")
    print(f"building Docker images for {len(specs)} instances ...")
    build_env_images(specs)
    print("done building images")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="GRPO RL on SWE-bench train (Docker-based rewards)"
    )
    # --- model / checkpointing ---
    ap.add_argument("--sft-ckpt", required=True, help="path to SFT (or prior RL) checkpoint")
    ap.add_argument("--run", default="codechat_2b_rl_swebench")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--keep-every", type=int, default=50)

    # --- GRPO hyper-parameters ---
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--group-size", type=int, default=4,
                    help="completions per prompt (each triggers reward eval)")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="diff patches need more tokens than code solutions")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-6,
                    help="lower than MBPP RL; SWE-bench signal is noisier")
    ap.add_argument("--kl-coef", type=float, default=0.02)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--tb-dir", default="runs/tb")

    # --- SWE-bench / Docker ---
    ap.add_argument(
        "--reward-mode",
        choices=["docker", "apply-only", "syntax"],
        default="docker",
        help="docker: full test eval (slow); "
             "apply-only: git apply check (medium); "
             "syntax: diff regex check (fast, no Docker)",
    )
    ap.add_argument("--docker-workers", type=int, default=2,
                    help="parallel Docker containers for group evaluation")
    ap.add_argument("--docker-timeout", type=int, default=300,
                    help="per-container timeout in seconds")
    ap.add_argument("--prepare-images", action="store_true",
                    help="build Docker images for train instances, then train")
    ap.add_argument("--train-limit", type=int, default=0,
                    help="use only first N train instances (0 = all)")
    args = ap.parse_args()

    seed_all(1337)
    assert DEVICE.type == "cuda"
    torch.set_float32_matmul_precision("high")

    # ---- Load policy + frozen reference ----
    print(f"loading checkpoint {args.sft_ckpt} ...")
    state = load_ckpt(args.sft_ckpt)
    cfg = GPTConfig(**state["cfg"])
    policy = GPT(cfg).to(DEVICE).to(COMPUTE_DTYPE)
    policy.load_state_dict(state["model"])
    ref = GPT(cfg).to(DEVICE).to(COMPUTE_DTYPE)
    ref.load_state_dict(state["model"])
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()
    print(f"  policy + ref loaded | depth={cfg.depth} n_embd={cfg.n_embd} "
          f"block_size={cfg.block_size}")

    # ---- Load SWE-bench train ----
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench", split="train")
    if args.train_limit > 0:
        ds = ds.select(range(min(args.train_limit, len(ds))))
    print(f"SWE-bench train: {len(ds)} instances")

    # ---- Optional: pre-build Docker images ----
    if args.prepare_images:
        _prepare_images(ds)
        if args.max_steps == 0:
            print("--max-steps 0: exiting after image preparation")
            return

    # ---- Reward environment ----
    reward_env = SWEBenchReward(
        mode=args.reward_mode,
        max_workers=args.docker_workers,
        timeout=args.docker_timeout,
    )
    print(f"reward: mode={args.reward_mode}  workers={args.docker_workers}  "
          f"timeout={args.docker_timeout}s")

    # ---- Optimizer + TensorBoard ----
    optim = build_optimizer(policy, lr=args.lr)
    ckpt_path = os.path.join(args.ckpt_dir, args.run, "latest.pt")
    tb_path = os.path.join(args.tb_dir, args.run)
    writer = SummaryWriter(log_dir=tb_path)
    print(f"tensorboard -> {tb_path}")
    t0 = time.time()

    # ---- Training loop ----
    n_skipped = 0
    for step in range(1, args.max_steps + 1):
        lr = cosine_lr(step, args.max_steps, args.lr, warmup=10)
        for g in optim.param_groups:
            g["lr"] = lr

        # Pick a random SWE-bench instance
        idx = int(torch.randint(0, len(ds), (1,)).item())
        instance = ds[idx]
        prompt = build_prompt(instance)
        prompt_ids = torch.tensor(
            [encode(prompt)], dtype=torch.long, device=DEVICE
        )
        prompt_len = prompt_ids.shape[1]

        # SWE-bench prompts can be very long — skip if they overflow context
        if prompt_len > cfg.block_size - args.max_new_tokens:
            n_skipped += 1
            if n_skipped % 50 == 0:
                print(f"  (skipped {n_skipped} instances so far — prompt too long)")
            continue

        # ---- Sample a group of completions ----
        step_t0 = time.time()
        rollouts = []   # list of full_ids tensors
        patches = []    # list of extracted diff strings
        for _ in range(args.group_size):
            new_ids, _lp, full_ids = sample_one(
                policy, prompt_ids, args.max_new_tokens,
                args.temperature, args.top_k,
            )
            text = decode(new_ids)
            if END_TAG in text:
                text = text.split(END_TAG)[0]
            patch = extract_diff(text)
            rollouts.append(full_ids)
            patches.append(patch)
        sample_time = time.time() - step_t0

        # ---- Compute rewards (potentially via Docker) ----
        reward_t0 = time.time()
        reward_results = reward_env.compute_rewards_batch(instance, patches)
        rewards_list = [r.reward for r in reward_results]
        reward_time = time.time() - reward_t0

        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=DEVICE)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        # ---- GRPO policy gradient + KL update ----
        policy.train()
        optim.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_pg = 0.0
        total_kl = 0.0
        for full_ids, a in zip(rollouts, adv):
            logp_pol = forward_logps(policy, full_ids, prompt_len)
            with torch.no_grad():
                logp_ref = forward_logps(ref, full_ids, prompt_len)
            # per-token KL approximation: logp_pol - logp_ref
            kl = (logp_pol - logp_ref).mean()
            # PG loss: -advantage * mean_logp
            pg = -a * logp_pol.mean()
            loss = pg + args.kl_coef * kl
            (loss / args.group_size).backward()
            total_loss += loss.item() / args.group_size
            total_pg += pg.item() / args.group_size
            total_kl += kl.item() / args.group_size
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), args.clip * 5
        )
        optim.step()

        # ---- TensorBoard logging ----
        n_applied = sum(1 for r in reward_results if r.patch_applies)
        n_tests_passed = sum(r.tests_passed for r in reward_results)
        n_nonempty = sum(1 for p in patches if p.strip())

        writer.add_scalar("swebrl/reward_mean", rewards.mean().item(), step)
        writer.add_scalar("swebrl/reward_max", rewards.max().item(), step)
        writer.add_scalar("swebrl/reward_min", rewards.min().item(), step)
        writer.add_scalar("swebrl/reward_std", rewards.std().item(), step)
        writer.add_scalar("swebrl/advantage_abs_mean", adv.abs().mean().item(), step)
        writer.add_scalar("swebrl/patches_nonempty", n_nonempty, step)
        writer.add_scalar("swebrl/patches_applied", n_applied, step)
        writer.add_scalar("swebrl/tests_passed_total", n_tests_passed, step)
        writer.add_scalar("swebrl/loss_total", total_loss, step)
        writer.add_scalar("swebrl/loss_pg", total_pg, step)
        writer.add_scalar("swebrl/kl", total_kl, step)
        writer.add_scalar("swebrl/lr", lr, step)
        writer.add_scalar("swebrl/grad_norm", float(grad_norm), step)
        writer.add_scalar("swebrl/prompt_len", prompt_len, step)
        writer.add_scalar("swebrl/sample_time_s", sample_time, step)
        writer.add_scalar("swebrl/reward_time_s", reward_time, step)
        writer.add_scalar("swebrl/step_time_s", time.time() - step_t0, step)
        writer.add_scalar("swebrl/elapsed_s", time.time() - t0, step)

        # ---- Console log ----
        if step % 2 == 0:
            iid = instance["instance_id"]
            errors = [r.error for r in reward_results if r.error]
            err_str = f" | err: {errors[0]}" if errors else ""
            print(
                f"swebrl {step:4d} | {iid} | "
                f"reward {rewards.mean().item():.3f} "
                f"(max {rewards.max().item():.2f}) | "
                f"applied {n_applied}/{args.group_size} | "
                f"tests {n_tests_passed} | "
                f"loss {total_loss:.4f} | lr {lr:.2e} | "
                f"gen {sample_time:.1f}s docker {reward_time:.1f}s"
                f"{err_str}"
            )

        # ---- Checkpointing ----
        if step % args.save_every == 0 or step == args.max_steps:
            save_ckpt(ckpt_path, policy, optim, step, cfg)
            step_path = os.path.join(
                args.ckpt_dir, args.run, f"step_{step:06d}.pt"
            )
            save_ckpt(step_path, policy, optim, step, cfg)
            # rotate: keep only multiples of --keep-every
            if args.keep_every > args.save_every:
                for p in glob.glob(
                    os.path.join(args.ckpt_dir, args.run, "step_*.pt")
                ):
                    base = os.path.basename(p)
                    try:
                        s = int(base[len("step_") :].split(".")[0])
                    except Exception:
                        continue
                    if s == step:
                        continue
                    if s % args.keep_every != 0:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
            print(f"  saved -> {ckpt_path} and {step_path}")

    # ---- Cleanup ----
    if n_skipped:
        print(f"total skipped: {n_skipped} instances (prompt too long for context)")
    elapsed = time.time() - t0
    print(f"training complete: {args.max_steps} steps in {elapsed/60:.1f} min")
    reward_env.close()
    writer.close()


if __name__ == "__main__":
    main()
