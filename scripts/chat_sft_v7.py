"""v7 joint SFT trainer — code + funcall in one pass, with per-domain eval.

Differences from chat_sft.py (intentionally kept as a sibling, not a refactor;
v* scripts are append-only history per CLAUDE.md):

  1. Uses SFTConvLoader (conversations format) so the dataloader can do the
     20% no-tools system-prompt injection that v6 lacked.

  2. Every --eval-every steps:
       - computes held-out supervised loss separately on eval_code.jsonl
         and eval_funcall.jsonl (each rank evaluates a shard, all-reduce mean)
       - logs sft/loss_code, sft/loss_funcall to TB
     This is the early-warning system v6 didn't have. v6's code ability
     silently rotted away over 5000+ steps because total loss kept improving
     (funcall was getting easier) while code-only loss was rising.

  3. Every --smoke-every steps, runs two inference smoke tests:
       - "Write a Python implementation of quicksort." (no system block)
         → expect 'def quicksort' in output, no <functioncall>
       - glaive-style system w/ get_weather schema + "Weather in Tokyo?"
         → expect <functioncall> w/ name=get_weather and a 'location' arg
     Logged as 0/1 to sft/smoke_code_pass and sft/smoke_funcall_pass.
     With FSDP, all ranks run the same forward (params all-gathered to identical
     state); rank 0 writes the result.

Same FSDP wrap, same optim, same ckpt code as chat_sft.py — duplicated
deliberately, see CLAUDE.md.
"""
import argparse
import functools
import json
import os
import re
import time
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from codechat.common import COMPUTE_DTYPE, seed_all
from codechat.gpt import GPT, GPTConfig, Block
from codechat.dataloader import SFTConvLoader
from codechat.optim import build_optimizer, cosine_lr
from codechat.checkpoint import save as save_ckpt, load as load_ckpt
from codechat.tokenizer import encode, decode, USER_TAG, ASSISTANT_TAG, END_TAG, EOT
from codechat.funcall_reward import _extract_functioncall_json, _parse_json_loose


SYSTEM_TAG = "<|system|>"

# Mock get_weather schema used in funcall smoke test. Shape mirrors what
# scripts/funcall_cli.py builds by default so SFT/inference stay aligned.
_WEATHER_TOOL_SYSTEM = (
    "You are a helpful assistant with access to the following functions. "
    "Use them if required -\n"
    + json.dumps({
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }, indent=2)
)


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


# ---------------------------------------------------------------------------
# Per-domain eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_loss(model, loader: SFTConvLoader, rank: int, world_size: int,
              is_dist: bool, device: torch.device) -> float:
    """Mean supervised CE over the held-out shard. Every rank MUST call
    model(...) the same number of times — FSDP FULL_SHARD all-gathers params
    inside forward, so a one-rank-at-a-time loop deadlocks the NCCL watchdog
    (was the v7 first-run crash). Round-robin: at iteration `it`, rank `r`
    handles example `it*world_size + r`; ranks past the end run a dummy pass
    whose loss is discarded so the collective stays balanced."""
    model.eval()
    total_loss = torch.zeros(1, device=device, dtype=torch.float32)
    total_count = torch.zeros(1, device=device, dtype=torch.float32)
    n = len(loader.examples)
    n_iter = (n + world_size - 1) // world_size
    for it in range(n_iter):
        idx = it * world_size + rank
        is_real = idx < n
        ex = loader.examples[idx if is_real else 0]
        ids, labels = loader._tokenize(ex["conversations"])
        x_row, y_row = loader._pack(ids, labels)
        x = torch.tensor([x_row], dtype=torch.long, device=device)
        y = torch.tensor([y_row], dtype=torch.long, device=device)
        _, loss = model(x, y)
        if is_real and torch.isfinite(loss):
            total_loss += loss.detach().float()
            total_count += 1.0
    if is_dist:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
    model.train()
    if total_count.item() == 0:
        return float("nan")
    return (total_loss / total_count).item()


# ---------------------------------------------------------------------------
# Smoke tests (FSDP-safe: all ranks forward identically, rank 0 logs)
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_generate(model, prompt_ids: torch.Tensor, max_new: int,
                    block_size: int, end_ids: set[int]) -> list[int]:
    """Argmax decoding. Identical across ranks because FSDP all-gathers params
    per layer in forward, so logits are the same on every rank."""
    model.eval()
    ids = prompt_ids
    new_tokens: list[int] = []
    for _ in range(max_new):
        cond = ids[:, -block_size:]
        logits, _ = model(cond)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        tok = int(nxt.item())
        new_tokens.append(tok)
        ids = torch.cat([ids, nxt], dim=1)
        if tok in end_ids and len(new_tokens) > 4:
            break
    model.train()
    return new_tokens


def _build_code_prompt() -> str:
    return f"{USER_TAG}\nWrite a Python implementation of quicksort.\n{END_TAG}\n{ASSISTANT_TAG}\n"


def _build_funcall_prompt() -> str:
    return (
        f"{SYSTEM_TAG}\n{_WEATHER_TOOL_SYSTEM}\n{END_TAG}\n"
        f"{USER_TAG}\nWeather in Tokyo?\n{END_TAG}\n{ASSISTANT_TAG}\n"
    )


_QUICKSORT_RE = re.compile(r"\bdef\s+(quicksort|partition|quick_sort)\b", re.I)


def smoke_code_pass(text: str) -> int:
    """Pass iff the reply contains a quicksort/partition def AND does NOT emit
    a <functioncall>. The negative half catches v5/v6's failure mode where
    'write quicksort' was answered with a bogus tool call."""
    if "<functioncall>" in text:
        return 0
    if _QUICKSORT_RE.search(text):
        return 1
    return 0


def smoke_funcall_pass(text: str) -> int:
    """Pass iff the reply contains a parseable <functioncall> with name
    'get_weather' and a 'location' arg. Doesn't check the location *value*
    because v6 showed value hallucination is a separate bug not addressable
    by this smoke."""
    blob = _extract_functioncall_json(text)
    if blob is None:
        return 0
    parsed = _parse_json_loose(blob)
    if not isinstance(parsed, dict):
        return 0
    if (parsed.get("name") or "").strip() != "get_weather":
        return 0
    args = parsed.get("arguments")
    if isinstance(args, str):
        args = _parse_json_loose(args)
    if not isinstance(args, dict):
        return 0
    return 1 if "location" in args else 0


def run_smoke(model, block_size: int, device: torch.device,
              max_new: int = 200) -> tuple[int, int, str, str]:
    end_ids = set(encode(END_TAG))
    # Code task
    code_prompt_ids = torch.tensor([encode(_build_code_prompt())],
                                   dtype=torch.long, device=device)
    code_new = greedy_generate(model, code_prompt_ids, max_new, block_size, end_ids)
    code_text = decode(code_new).split(END_TAG)[0].strip()
    code_pass = smoke_code_pass(code_text)
    # Funcall task
    func_prompt_ids = torch.tensor([encode(_build_funcall_prompt())],
                                   dtype=torch.long, device=device)
    func_new = greedy_generate(model, func_prompt_ids, max_new, block_size, end_ids)
    func_text = decode(func_new).split(END_TAG)[0].strip()
    func_pass = smoke_funcall_pass(func_text)
    return code_pass, func_pass, code_text, func_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--data-dir", default="data/sft_v7",
                    help="dir containing train.jsonl + eval_code.jsonl + "
                         "eval_funcall.jsonl (output of scripts.prepare_sft_v7)")
    ap.add_argument("--device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=200,
                    help="per-domain held-out loss eval cadence")
    ap.add_argument("--smoke-every", type=int, default=500,
                    help="quicksort+weather generation smoke cadence")
    ap.add_argument("--smoke-max-new", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--system-inject-ratio", type=float, default=0.20,
                    help="probability of injecting a no-tools system prompt "
                         "for non-toolcall samples; minimind default")
    ap.add_argument("--run-name", "--run", dest="run", default="codechat_8b_sft_v7")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--tb-dir", default="runs/tb")
    args = ap.parse_args()

    is_dist, local_rank, rank, world_size = setup_distributed()
    is_master = (rank == 0)

    seed_all(1337 + rank)
    device = torch.device("cuda", local_rank) if is_dist else torch.device("cuda")
    assert device.type == "cuda"
    torch.set_float32_matmul_precision("high")

    state = load_ckpt(args.base_ckpt, map_location="cpu")
    cfg = GPTConfig(**state["cfg"])
    model = GPT(cfg)
    model.load_state_dict(state["model"])
    del state
    if is_master:
        print(f"loaded base ckpt {args.base_ckpt}  (depth={cfg.depth}, "
              f"block_size={cfg.block_size})")

    model = model.to(device).to(COMPUTE_DTYPE)
    if is_dist:
        model = wrap_fsdp(model)

    train_loader = SFTConvLoader(
        os.path.join(args.data_dir, "train.jsonl"),
        batch_size=args.device_batch_size,
        block_size=cfg.block_size,
        device=device,
        rank=rank,
        world_size=world_size,
        seed=1337,
        system_inject_ratio=args.system_inject_ratio,
    )
    eval_code = SFTConvLoader(
        os.path.join(args.data_dir, "eval_code.jsonl"),
        batch_size=1, block_size=cfg.block_size, device=device,
        rank=rank, world_size=world_size, seed=1337,
        system_inject_ratio=0.0,  # eval is deterministic
    )
    eval_funcall = SFTConvLoader(
        os.path.join(args.data_dir, "eval_funcall.jsonl"),
        batch_size=1, block_size=cfg.block_size, device=device,
        rank=rank, world_size=world_size, seed=1337,
        system_inject_ratio=0.0,
    )
    if is_master:
        print(f"train: {len(train_loader)} rows  "
              f"eval_code: {len(eval_code)} rows  "
              f"eval_funcall: {len(eval_funcall)} rows")

    optim = build_optimizer(model, lr=args.lr)

    ckpt_path = os.path.join(args.ckpt_dir, args.run, "latest.pt")
    tb_path = os.path.join(args.tb_dir, args.run)
    writer = SummaryWriter(log_dir=tb_path) if is_master else None
    if is_master:
        print(f"tensorboard -> {tb_path}")

    t0 = time.time()
    model.train()
    for step in range(1, args.max_steps + 1):
        lr = cosine_lr(step, args.max_steps, args.lr, warmup=args.warmup)
        for g in optim.param_groups:
            g["lr"] = lr
        optim.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(args.grad_accum):
            x, y = train_loader.next_batch()
            if is_dist and micro < args.grad_accum - 1 and hasattr(model, "no_sync"):
                with model.no_sync():
                    _, loss = model(x, y)
                    (loss / args.grad_accum).backward()
            else:
                _, loss = model(x, y)
                (loss / args.grad_accum).backward()
            loss_accum += loss.item() / args.grad_accum

        if is_dist and hasattr(model, "clip_grad_norm_"):
            grad_norm = model.clip_grad_norm_(1.0)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if is_master:
            writer.add_scalar("sft/loss", loss_accum, step)
            writer.add_scalar("sft/lr", lr, step)
            writer.add_scalar("sft/grad_norm", float(grad_norm), step)
            writer.add_scalar("sft/elapsed_s", time.time() - t0, step)

        if step % 20 == 0 and is_master:
            print(f"sft step {step:5d} | loss {loss_accum:.4f} | lr {lr:.2e} "
                  f"| {time.time()-t0:.0f}s")

        # Per-domain held-out loss
        if step % args.eval_every == 0 or step == args.max_steps:
            t_eval = time.time()
            l_code = eval_loss(model, eval_code, rank, world_size, is_dist, device)
            l_func = eval_loss(model, eval_funcall, rank, world_size, is_dist, device)
            if is_master:
                writer.add_scalar("sft/loss_code", l_code, step)
                writer.add_scalar("sft/loss_funcall", l_func, step)
                print(f"  [eval step={step}] loss_code={l_code:.4f} "
                      f"loss_funcall={l_func:.4f}  ({time.time()-t_eval:.1f}s)")

        # Inference smoke (greedy gen, identical on all ranks, master logs)
        if step % args.smoke_every == 0 or step == args.max_steps:
            t_smoke = time.time()
            code_pass, func_pass, code_text, func_text = run_smoke(
                model, cfg.block_size, device, max_new=args.smoke_max_new)
            if is_master:
                writer.add_scalar("sft/smoke_code_pass", code_pass, step)
                writer.add_scalar("sft/smoke_funcall_pass", func_pass, step)
                tag_c = "PASS" if code_pass else "FAIL"
                tag_f = "PASS" if func_pass else "FAIL"
                print(f"  [smoke step={step}] code={tag_c} funcall={tag_f}  "
                      f"({time.time()-t_smoke:.1f}s)")
                # Truncated text for the log so a regression is debuggable
                # without scrolling through full TensorBoard text panes.
                print(f"    code: {code_text[:160]!r}")
                print(f"    func: {func_text[:160]!r}")

        if step % args.save_every == 0 or step == args.max_steps:
            save_ckpt(ckpt_path, model, optim, step, cfg)
            if is_master:
                print(f"  saved -> {ckpt_path}")

    if is_master:
        writer.close()
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
