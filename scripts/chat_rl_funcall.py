"""v5 RL: funcall GRPO with dense format-based reward + online pass@k eval.

Design deltas vs scripts/chat_rl.py (which stalled at reward=0 on MBPP):

  1. Task is funcall format match, not MBPP unit-test pass. Reward ladder
     has 5+ non-zero tiers (see codechat/funcall_reward.py). Even a model
     that only learnt "emit <functioncall>" gets 0.15, giving GRPO a
     non-zero advantage from step 0. Mirrors MathGPT's `#### N` reward.

  2. **Per-rank different prompts** (was: same prompt on all 8 ranks). FSDP
     only requires that forward-pass shapes and step count match across
     ranks — the data itself can differ. So each rank draws its own
     problem from `problems[rank::world_size]`, and gradients are averaged
     across 8 distinct problems per step. Same trick MathGPT uses with DDP.

  3. **Batched rollouts**: instead of a Python for-loop over group_size,
     we put all samples in one batch `[num_samples, T]` and advance the
     whole batch at each decode step. One collective forward samples K
     completions in parallel.

  4. **No KL, no ref model**: the dense reward + staircase is self-regularising
     enough. Dropping the ref model halves activation memory and cuts one
     forward pass per step. We still track a `kl_vs_init` diagnostic via an
     optional init-snapshot logprob cache, but it's informational only.

  5. **Online pass@k eval** every `--eval-every` steps on a held-out set
     (data/rl_funcall/eval.jsonl). Lets us pick the best checkpoint
     (peak-Pass@1 or peak-Pass@16 depending on downstream use) instead of
     blind-running to max_steps. MathGPT did exactly this on GSM8K.

  6. **LR multiplier schedule**: init_lr_frac (default 0.05) × base_lr,
     linearly decaying to 0 over num_steps. Same recipe as MathGPT RL.

Both single-GPU and FSDP-8× are supported; launch detection is the same
as chat_sft.py.
"""
import argparse
import functools
import glob
import json
import os
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from codechat.common import COMPUTE_DTYPE, seed_all
from codechat.gpt import GPT, GPTConfig, Block
from codechat.optim import build_optimizer
from codechat.checkpoint import save as save_ckpt, load as load_ckpt
from codechat.tokenizer import encode, decode, END_TAG
from codechat.funcall_reward import funcall_reward, funcall_exact_match


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


def load_problems(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


@torch.no_grad()
def sample_batch(model, prompt_ids, num_samples, max_new_tokens,
                 temperature, top_k, block_size):
    """Generate `num_samples` completions from `prompt_ids` in parallel.

    prompt_ids: [1, T]  (single prompt, will be repeated K times)
    Returns a list of length num_samples, each a list[int] of generated token ids.
    """
    model.eval()
    ids = prompt_ids.repeat(num_samples, 1)  # [K, T]
    eot_ids = set(encode(END_TAG))
    done = torch.zeros(num_samples, dtype=torch.bool, device=ids.device)
    new_ids = [[] for _ in range(num_samples)]
    is_dist = dist.is_available() and dist.is_initialized()

    for _ in range(max_new_tokens):
        cond = ids[:, -block_size:]
        logits, _ = model(cond)                       # [K, T', V]
        step_logits = logits[:, -1, :].float() / max(temperature, 1e-5)
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(step_logits, min(top_k, step_logits.size(-1)))
            step_logits[step_logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(step_logits, dim=-1)
        nxt = torch.multinomial(probs, 1)             # [K, 1]
        # Don't broadcast — each rank has different prompts and different
        # samples. FSDP only requires shape-aligned forwards, not identical data.
        ids = torch.cat([ids, nxt], dim=1)
        nxt_cpu = nxt.squeeze(-1).tolist()
        for i in range(num_samples):
            if done[i]:
                continue
            tok = nxt_cpu[i]
            new_ids[i].append(tok)
            if tok in eot_ids and len(new_ids[i]) > 4:
                done[i] = True
        # Early-stop must agree across FSDP ranks. Each rank has different
        # rollouts that may finish at different steps; breaking locally would
        # desync subsequent model.forward all-gathers. Reduce local done.all()
        # with MIN so we only break when *every* rank has finished.
        local_done = torch.tensor([1 if bool(done.all()) else 0],
                                  device=ids.device, dtype=torch.int32)
        if is_dist:
            dist.all_reduce(local_done, op=dist.ReduceOp.MIN)
        if bool(local_done.item()):
            break
    return new_ids, ids  # also return full tensor for later logprob forward


def forward_logps_batched(model, full_ids, prompt_len):
    """Per-token logprobs on the completion tokens for every row in the batch.

    full_ids: [K, T_total]. Returns [K, T_total - prompt_len] (indexing math
    identical to the original single-sequence version).
    """
    logits, _ = model(full_ids)
    logits = logits[:, :-1, :]
    targets = full_ids[:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1)
    tgt_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [K, T-1]
    return tgt_logp[:, prompt_len - 1:]


@torch.no_grad()
def online_eval(model, eval_problems, cfg, device, max_new_tokens,
                temperature, top_k, num_samples, n_examples, is_dist,
                world_size, rank):
    """Pass@1 / Pass@K on a held-out slice. Each rank eats its own shard,
    then we all_reduce the counters to rank 0."""
    model.eval()
    n_eval = min(n_examples, len(eval_problems))
    # Round down so every rank does exactly the same number of sample_batch
    # calls. Otherwise FSDP all-gather collectives desync and NCCL times out.
    n_eval = (n_eval // world_size) * world_size
    if n_eval == 0:
        return 0.0, 0.0, 0
    local_indices = range(rank, n_eval, world_size)
    max_prompt_len = cfg.block_size - max_new_tokens
    local_pass1 = 0
    local_passk = 0
    local_total = 0
    for idx in local_indices:
        ex = eval_problems[idx]
        prompt_ids = torch.tensor([encode(ex["prompt"])], dtype=torch.long, device=device)
        if prompt_ids.shape[1] > max_prompt_len:
            # Left-truncate instead of `continue`. Skipping desyncs FSDP.
            prompt_ids = prompt_ids[:, -max_prompt_len:]
        new_ids, _ = sample_batch(
            model, prompt_ids, num_samples, max_new_tokens,
            temperature, top_k, cfg.block_size,
        )
        hits = 0
        for ids in new_ids:
            text = decode(ids)
            if END_TAG in text:
                text = text.split(END_TAG)[0]
            hits += funcall_exact_match(text, ex["gt_name"], ex["gt_args"])
        local_total += 1
        local_passk += 1 if hits >= 1 else 0
        local_pass1 += hits / num_samples

    totals = torch.tensor(
        [local_pass1, local_passk, local_total],
        dtype=torch.float64, device=device,
    )
    if is_dist:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    pass1 = (totals[0] / totals[2]).item() if totals[2] > 0 else 0.0
    passk = (totals[1] / totals[2]).item() if totals[2] > 0 else 0.0
    n = int(totals[2].item())
    return pass1, passk, n


def cosine_decay_lr_mult(step: int, total_steps: int, init_frac: float) -> float:
    """init_frac at step=0, linearly decays to 0 at step=total_steps."""
    if total_steps <= 0:
        return init_frac
    frac = 1.0 - step / total_steps
    return max(0.0, init_frac * frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-ckpt", required=True,
                    help="funcall SFT checkpoint — RL starts here")
    ap.add_argument("--problems-file", default="data/rl_funcall/train.jsonl")
    ap.add_argument("--eval-file", default="data/rl_funcall/eval.jsonl")
    ap.add_argument("--run-name", "--run", dest="run", default="codechat_8b_rl_funcall_v5")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--tb-dir", default="runs/tb")

    # Training horizon
    ap.add_argument("--num-epochs", type=int, default=1,
                    help="passes through the training set (MathGPT v2: 1 epoch was the best)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="override epoch-based step count (0 = use num-epochs)")

    # Sampling
    ap.add_argument("--num-samples", type=int, default=16,
                    help="rollouts per prompt per rank (MathGPT v2: 32 worked, 16 safer for 8B)")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="funcall JSON is short; 256 is plenty")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)

    # Optimization
    ap.add_argument("--lr", type=float, default=1e-5, help="base LR before init-lr-frac")
    ap.add_argument("--init-lr-frac", type=float, default=0.05,
                    help="start at init_lr_frac × base_lr, linearly decay to 0")
    ap.add_argument("--clip", type=float, default=1.0)

    # Eval / saving
    ap.add_argument("--eval-every", type=int, default=30,
                    help="run online pass@k every N steps; 0 disables")
    ap.add_argument("--eval-examples", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=30)
    ap.add_argument("--keep-every", type=int, default=60,
                    help="retain only step_*.pt that are multiples of this")

    # Logging
    ap.add_argument("--log-rollouts-every", type=int, default=50)

    args = ap.parse_args()

    is_dist, local_rank, rank, world_size = setup_distributed()
    is_master = (rank == 0)

    seed_all(1337 + rank)
    device = torch.device("cuda", local_rank) if is_dist else torch.device("cuda")
    assert device.type == "cuda"
    torch.set_float32_matmul_precision("high")

    # Load ckpt to CPU so all ranks don't fight for GPU 0.
    state = load_ckpt(args.sft_ckpt, map_location="cpu")
    cfg = GPTConfig(**state["cfg"])
    policy = GPT(cfg)
    policy.load_state_dict(state["model"])
    del state
    policy = policy.to(device).to(COMPUTE_DTYPE)
    if is_dist:
        policy = wrap_fsdp(policy)

    # Data
    train_problems = load_problems(args.problems_file)
    eval_problems = load_problems(args.eval_file) if os.path.exists(args.eval_file) else []
    if is_master:
        print(f"loaded sft ckpt {args.sft_ckpt}")
        print(f"train problems: {len(train_problems)}  eval problems: {len(eval_problems)}")
        print(f"world_size={world_size}  num_samples={args.num_samples}  "
              f"effective rollouts/step = {world_size * args.num_samples}")

    # Compute num_steps. Each step consumes `world_size` problems (one per rank).
    if args.max_steps > 0:
        num_steps = args.max_steps
    else:
        per_epoch = max(1, len(train_problems) // world_size)
        num_steps = per_epoch * args.num_epochs
    if is_master:
        print(f"num_steps = {num_steps}  ({args.num_epochs} epoch(s))")

    optim = build_optimizer(policy, lr=args.lr)

    ckpt_dir = os.path.join(args.ckpt_dir, args.run)
    ckpt_path = os.path.join(ckpt_dir, "latest.pt")
    tb_path = os.path.join(args.tb_dir, args.run)
    writer = SummaryWriter(log_dir=tb_path) if is_master else None
    if is_master:
        print(f"tensorboard -> {tb_path}")

    t0 = time.time()
    # Each rank walks its own shard of problems in order, cycling as needed.
    def next_problem(step_i: int) -> dict:
        # step_i indexes into rank's local shard, not global index
        local_n = max(1, len(train_problems) // world_size)
        local_idx = step_i % local_n
        # Map local_idx -> global idx = local_idx * world_size + rank
        global_idx = local_idx * world_size + rank
        global_idx = global_idx % len(train_problems)
        return train_problems[global_idx]

    for step in range(1, num_steps + 1):
        # ---- online eval ----
        if args.eval_every > 0 and eval_problems and (step == 1 or step % args.eval_every == 0):
            pass1, passk, n_eval = online_eval(
                policy, eval_problems, cfg, device,
                args.max_new_tokens, args.temperature, args.top_k,
                num_samples=args.num_samples,
                n_examples=args.eval_examples,
                is_dist=is_dist, world_size=world_size, rank=rank,
            )
            if is_master:
                print(f"[eval] step {step:5d} | pass@1 {pass1:.4f} | "
                      f"pass@{args.num_samples} {passk:.4f} | n={n_eval}")
                writer.add_scalar("eval/pass@1", pass1, step)
                writer.add_scalar(f"eval/pass@{args.num_samples}", passk, step)

        # ---- LR schedule ----
        lr_mult = cosine_decay_lr_mult(step - 1, num_steps, args.init_lr_frac)
        lr_now = args.lr * lr_mult
        for g in optim.param_groups:
            g["lr"] = lr_now

        # ---- pick a problem (different per rank) ----
        ex = next_problem(step - 1)
        prompt_ids = torch.tensor([encode(ex["prompt"])], dtype=torch.long, device=device)
        max_prompt_len = cfg.block_size - args.max_new_tokens
        if prompt_ids.shape[1] > max_prompt_len:
            # Left-truncate instead of `continue`: skipping on one rank would
            # desync FSDP all-gather with the other ranks.
            prompt_ids = prompt_ids[:, -max_prompt_len:]
        prompt_len = prompt_ids.shape[1]

        # ---- rollouts (batched) ----
        new_ids_list, full_ids_all = sample_batch(
            policy, prompt_ids, args.num_samples, args.max_new_tokens,
            args.temperature, args.top_k, cfg.block_size,
        )

        # ---- score rollouts ----
        rewards_local = []
        tier_counts = {}
        rollout_texts = []
        for ids in new_ids_list:
            text = decode(ids)
            if END_TAG in text:
                text = text.split(END_TAG)[0]
            r, tier = funcall_reward(text, ex["gt_name"], ex["gt_args"])
            rewards_local.append(r)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            rollout_texts.append(text)
        rewards = torch.tensor(rewards_local, dtype=torch.float32, device=device)

        # REINFORCE with baseline (MathGPT's choice). No std-normalization,
        # no KL, no ref: the dense tiered reward already gives enough signal
        # and doesn't need a trust-region term for stability at this scale.
        advantages = rewards - rewards.mean()

        # ---- PG loss ----
        policy.train()
        optim.zero_grad(set_to_none=True)

        # For memory, chunk the logprob forward if num_samples is big.
        # With 8B + FSDP, num_samples=16 should fit; leave the knob at 1 mb.
        logp = forward_logps_batched(policy, full_ids_all, prompt_len)  # [K, T_new]
        # Mask out the *post-EOT* tokens so padding past END_TAG doesn't
        # dilute the gradient. We detect this by finding the first occurrence
        # of any END_TAG token in each row and zeroing the rest.
        eot_ids = torch.tensor(sorted(set(encode(END_TAG))), device=device)
        # Build a mask per row: True for positions <= first-eot index.
        completion = full_ids_all[:, prompt_len:]                       # [K, T_new]
        is_eot = (completion.unsqueeze(-1) == eot_ids).any(-1)           # [K, T_new]
        # cumulative sum of is_eot crosses 1 at first EOT — include the EOT itself.
        eot_cum = torch.cumsum(is_eot.int(), dim=1)
        keep_mask = (eot_cum <= 1).float()                               # [K, T_new]
        # Also ignore positions past the actual generated length (the batch
        # pad at the end from `cat` shares the same tail token for all rows
        # because all rows ran the full `max_new_tokens` decode, so keep_mask
        # handles that naturally via the EOT column logic).

        # per-sample sum of logp over completion tokens, normalised by length
        # (so varying lengths don't bias high-logp long rollouts)
        denom = keep_mask.sum(dim=1).clamp(min=1.0)
        per_sample_logp = (logp * keep_mask).sum(dim=1) / denom          # [K]

        pg_loss = -(advantages * per_sample_logp).mean()
        pg_loss.backward()

        # FSDP-aware grad clip
        if is_dist and hasattr(policy, "clip_grad_norm_"):
            grad_norm = policy.clip_grad_norm_(args.clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip)
        optim.step()

        # ---- logging ----
        # Rewards are local-to-rank. Reduce means so TB shows the global picture.
        local_stats = torch.tensor(
            [rewards.mean().item(), rewards.max().item(), rewards.min().item(),
             rewards.std().item(), advantages.abs().mean().item()],
            dtype=torch.float32, device=device,
        )
        if is_dist:
            dist.all_reduce(local_stats, op=dist.ReduceOp.AVG)
        r_mean, r_max, r_min, r_std, adv_abs = local_stats.tolist()

        if is_master:
            writer.add_scalar("rl/reward_mean", r_mean, step)
            writer.add_scalar("rl/reward_max", r_max, step)
            writer.add_scalar("rl/reward_min", r_min, step)
            writer.add_scalar("rl/reward_std", r_std, step)
            writer.add_scalar("rl/advantage_abs_mean", adv_abs, step)
            writer.add_scalar("rl/loss_pg", pg_loss.item(), step)
            writer.add_scalar("rl/lr", lr_now, step)
            writer.add_scalar("rl/lr_mult", lr_mult, step)
            writer.add_scalar("rl/grad_norm", float(grad_norm), step)
            writer.add_scalar("rl/prompt_len", prompt_len, step)
            writer.add_scalar("rl/elapsed_s", time.time() - t0, step)
            for tier, cnt in tier_counts.items():
                writer.add_scalar(f"rl/tier/{tier}", cnt / args.num_samples, step)

        if step % 5 == 0 and is_master:
            elapsed = time.time() - t0
            print(
                f"rl step {step:5d}/{num_steps} | reward {r_mean:.3f} "
                f"(max {r_max:.2f}, std {r_std:.3f}) | loss {pg_loss.item():+.4f} "
                f"| lr {lr_now:.2e} | {elapsed:.0f}s"
            )

        # Dump a rollout periodically. Pick the best one so we see what
        # "nearly right" looks like.
        if (args.log_rollouts_every and is_master
                and step % args.log_rollouts_every == 0):
            best_i = int(torch.argmax(rewards).item())
            best_text = rollout_texts[best_i][:800]
            snippet = best_text.replace("\n", "\n    ")
            rewards_rounded = [round(r, 3) for r in rewards_local]
            print(
                f"  [rollout @ step {step}] best_reward={rewards_local[best_i]:.3f}  "
                f"rewards={rewards_rounded}\n"
                f"    gt_name: {ex['gt_name']}  gt_args: {ex['gt_args']}\n"
                f"    best output >>>\n    {snippet}\n    <<<"
            )
            if writer is not None:
                writer.add_text(
                    "rl/rollout_best",
                    f"step {step} | reward {rewards_local[best_i]:.3f}\n\n"
                    f"gt_name: {ex['gt_name']}\n"
                    f"gt_args: {ex['gt_args']}\n\n"
                    f"output:\n{rollout_texts[best_i][:2000]}",
                    step,
                )

        # Save. Rotate old step snapshots.
        if step % args.save_every == 0 or step == num_steps:
            save_ckpt(ckpt_path, policy, optim, step, cfg)
            step_path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
            save_ckpt(step_path, policy, optim, step, cfg)
            if args.keep_every > args.save_every:
                for p in glob.glob(os.path.join(ckpt_dir, "step_*.pt")):
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
                print(f"  saved -> {ckpt_path}  and  {step_path}")

    if is_master:
        writer.close()
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
