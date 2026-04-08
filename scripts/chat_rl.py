"""GRPO-style RL on MBPP with executable rewards.

For each prompt we sample G completions from the current policy, run the
reference test asserts in a subprocess, and use (reward - group_mean) /
(group_std + eps) as the advantage. Loss is the standard PG loss with a
KL penalty toward a frozen reference model to keep the policy near the SFT
checkpoint.

Single-GPU, bf16, A800 80GB friendly.
"""
import argparse
import os
import time
import copy
import torch
import torch.nn.functional as F

from codechat.common import DEVICE, COMPUTE_DTYPE, seed_all
from codechat.gpt import GPT, GPTConfig
from codechat.optim import build_optimizer, cosine_lr
from codechat.checkpoint import save as save_ckpt, load as load_ckpt
from codechat.tokenizer import encode, decode, USER_TAG, ASSISTANT_TAG, END_TAG
from codechat.execution import run_with_tests, extract_code


def build_prompt(problem: str) -> str:
    user = (
        "Solve the following Python problem. Return ONLY a Python code block.\n\n"
        + problem
    )
    return f"{USER_TAG}\n{user}\n{END_TAG}\n{ASSISTANT_TAG}\n"


def sample_one(model: GPT, prompt_ids: torch.Tensor, max_new: int, temperature: float, top_k: int):
    """Sample a single completion. Returns (token_ids [T_new], logprobs [T_new])."""
    model.eval()
    ids = prompt_ids.clone()
    new_ids, new_logps = [], []
    eot_str_ids = set(encode(END_TAG))
    with torch.no_grad():
        for _ in range(max_new):
            cond = ids[:, -model.cfg.block_size:]
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
    ap.add_argument("--run", default="codechat_d20_rl")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--group-size", type=int, default=4, help="completions per prompt")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-coef", type=float, default=0.02)
    ap.add_argument("--clip", type=float, default=0.2)
    args = ap.parse_args()

    seed_all(1337)
    assert DEVICE.type == "cuda"
    torch.set_float32_matmul_precision("high")

    state = load_ckpt(args.sft_ckpt)
    cfg = GPTConfig(**state["cfg"])
    policy = GPT(cfg).to(DEVICE).to(COMPUTE_DTYPE)
    policy.load_state_dict(state["model"])
    ref = GPT(cfg).to(DEVICE).to(COMPUTE_DTYPE)
    ref.load_state_dict(state["model"])
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()

    from datasets import load_dataset
    mbpp = load_dataset("mbpp", "sanitized", split="train")
    print(f"loaded {len(mbpp)} MBPP problems")

    optim = build_optimizer(policy, lr=args.lr)
    ckpt_path = os.path.join(args.ckpt_dir, args.run, "latest.pt")
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        lr = cosine_lr(step, args.max_steps, args.lr, warmup=20)
        for g in optim.param_groups:
            g["lr"] = lr

        ex = mbpp[int(torch.randint(0, len(mbpp), (1,)).item())]
        problem = ex["prompt"]
        tests = ex["test_list"]
        prompt = build_prompt(problem)
        prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
        prompt_len = prompt_ids.shape[1]
        if prompt_len > cfg.block_size - args.max_new_tokens:
            continue

        # ---- sample a group of completions and score ----
        rollouts = []  # list of (full_ids, reward)
        for _ in range(args.group_size):
            new_ids, _old_lp, full_ids = sample_one(
                policy, prompt_ids, args.max_new_tokens, args.temperature, args.top_k
            )
            text = decode(new_ids)
            if END_TAG in text:
                text = text.split(END_TAG)[0]
            code = extract_code(text)
            reward = run_with_tests(code, tests)
            rollouts.append((full_ids, reward))

        rewards = torch.tensor([r for _, r in rollouts], dtype=torch.float32, device=DEVICE)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        # ---- PG + KL update ----
        policy.train()
        optim.zero_grad(set_to_none=True)
        total_loss = 0.0
        for (full_ids, _), a in zip(rollouts, adv):
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
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip * 5)
        optim.step()

        if step % 5 == 0:
            print(
                f"rl step {step:5d} | reward {rewards.mean().item():.3f} "
                f"(max {rewards.max().item():.2f}) | loss {total_loss:.4f} "
                f"| lr {lr:.2e} | {time.time()-t0:.0f}s"
            )
        if step % 200 == 0 or step == args.max_steps:
            save_ckpt(ckpt_path, policy, optim, step, cfg)
            print(f"  saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
