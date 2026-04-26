# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Behavioral guidelines

These bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think before coding — don't assume, don't hide confusion, surface tradeoffs

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

This matters here specifically because: training-script knobs (`SKIP_TO`, `FORCE_*`, `--run-name`) are easy to misread, and a wrong guess can wipe a multi-hour run's Tensorboard history or restart from stage 1. Confirm before invoking shell pipelines.

### 2. Simplicity first — minimum code that solves the problem

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Senior-engineer test: would they say this is overcomplicated? If yes, simplify. Note the existing scripts deliberately duplicate `setup_distributed`/`wrap_fsdp` across `base_train.py` / `chat_sft.py` / `chat_rl.py` rather than factoring them out — match that style; don't introduce a shared helper module unless asked.

### 3. Surgical changes — touch only what you must

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/vars/functions YOUR changes orphaned. Leave pre-existing dead code alone unless asked.

Every changed line should trace directly to the user's request. Versioned pipelines (`v2`, `v3`, `v5`, `v6`) are append-only history — do not edit older `train_a800_x8_v*.sh` to "fix" them; the reports in `reports/` reference their exact behavior.

### 4. Goal-driven execution — define success, loop until verified

Convert tasks into verifiable goals before coding:
- "Add validation" → write a test for invalid inputs, then make it pass.
- "Fix the bug" → write a test that reproduces it, then make it pass.
- "Refactor X" → tests pass before and after.

For multi-step work, state a brief plan with a verification check per step:

```
1. [step] → verify: [check]
2. [step] → verify: [check]
```

For training/eval changes specifically: the verification is usually a smoke run (`MAX_EXAMPLES=2000`, `--max-steps 200`, or a single `chat_cli`/`funcall_cli` invocation), not just `python -m py_compile`. Type-checking and parsing don't catch loss masking, FSDP wrap, or tokenizer-tag bugs.

---

## What this repo is

CodeChat is a from-scratch nanochat-style transformer LM (decoder-only, GPT-2 BPE tokenizer, weight-tied head) trained for Python coding and tool/function calling. There is no HuggingFace `transformers` dependency — `codechat/gpt.py` is the model. Two main hardware paths share the same code, distinguished only at launch time:

- **Single A800 80GB → preset `2b`** (`runs/train_a800.sh`, `train_a800_v2.sh`)
- **8× A800 80GB → preset `8b`** (`runs/train_a800_x8*.sh`), wrapped in **FSDP `FULL_SHARD`** (= ZeRO-3) because the 8B fp32 AdamW state alone is ~96GB, beyond a single 80GB card

Distributed mode is auto-detected from `LOCAL_RANK`/`RANK`/`WORLD_SIZE` (set by `torchrun`). The same scripts (`base_train.py`, `chat_sft.py`, `chat_rl.py`) work in both modes. See `README.md` for the full 8B FSDP rationale and `docs/README_A800_x1.md` for the single-GPU 2B path.

## Common commands

The 8-GPU host has no writable system site-packages. The shell scripts bootstrap a `.venv_train` venv built with `--system-site-packages` so it inherits the host's pre-installed `torch` (do not `pip install torch` into the venv). On a single-GPU box you can install `requirements.txt` normally.

```bash
# 8-GPU end-to-end pipelines (auto-create .venv_train on first run; idempotent)
bash runs/train_a800_x8.sh                     # base pretrain + SFT (+ legacy MBPP RL)
bash runs/train_a800_x8_v2_funcall.sh          # funcall SFT + "rescued" MBPP RL (3-stage)
bash runs/train_a800_x8_v6.sh                  # joint code+funcall SFT, then funcall RL
SKIP_TO=N bash runs/train_a800_x8_v6.sh        # resume any pipeline from stage N
bash runs/launch_v6_bg.sh [N|status|tail|stop] # nohup wrapper around v6, logs in runs/logs/

# Direct invocation (after venv is built). NOTE: use --run-name, not --run, under torchrun.
./.venv_train/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node=8 \
    -m scripts.base_train --data-dir data/pretrain --preset 8b \
    --device-batch-size 1 --grad-accum 8 --max-steps 30000 \
    --lr 1.5e-4 --warmup 1000 --run-name codechat_8b

# Single-GPU 2B
python -m scripts.base_train --preset 2b --run-name codechat_2b ...

# Inference / smoke tests
python -m scripts.chat_cli --ckpt checkpoints/<run>/latest.pt --user "write quicksort"
python -m scripts.funcall_cli --ckpt checkpoints/<run>/latest.pt --executors my_tools.py \
    --user "Weather in Tokyo?"

# Eval
python -m scripts.eval_mbpp_pass_at_k --ckpt <ckpt> --k 8           # pass@k diagnostic + VERDICT
python -m scripts.eval_funcall --ckpt <ckpt>                        # funcall exact-match
python -m scripts.filter_mbpp_by_passrate --ckpt <ckpt> --out ...   # keep pass_rate ∈ [0.05, 0.95]

# Tensorboard (all runs land under runs/tb/<run-name>/)
./.venv_train/bin/tensorboard --logdir runs/tb
```

There is no test suite, lint config, or `pyproject` build target beyond the package metadata in `pyproject.toml`. There is no `pip install -e .`; the shell scripts set `PYTHONPATH=$REPO_ROOT` so `import codechat.*` resolves.

## Critical foot-guns (read before invoking torchrun)

- **`--run-name`, never `--run`.** Under `torchrun`, argparse sees `--run` as an ambiguous prefix of torchrun's own `--run-path` and aborts before forwarding args. The training scripts accept both, but only `--run-name` is safe in 8-GPU launches.
- **Save-time OOM happens on rank 0's CPU**, not GPU. `codechat/checkpoint.save()` gathers a full FSDP state dict to rank-0 CPU (`offload_to_cpu=True, rank0_only=True`) so the on-disk file is identical to a single-GPU checkpoint and `chat_cli`/`funcall_cli` can load it without FSDP. Host RAM must fit ~32GB for 8B fp32. Optimizer state is intentionally NOT saved (FSDP optim sharded-save is non-trivial; pretrain restarts the optimizer anyway).
- **2B and 8B checkpoints are not interchangeable.** Different `depth`/`n_embd` → `load_state_dict(strict=True)` errors. There is no built-in upscaling loader.
- **Activation checkpointing must stay on for 8B.** `cfg.grad_checkpoint=True` is required to fit even with FSDP; `device_batch_size=1`, `grad_accum=8` is the tested combo.
- **FSDP requires `use_orig_params=True`.** Without it, `optim.py`'s `p.ndim>=2` decay/no-decay split sees flat 1-D params and silently breaks weight decay grouping.

## Code layout / what to read when

`codechat/` is the library; `scripts/` is the entry points; `runs/` is the shell glue.

- `codechat/gpt.py` — model + `PRESETS` (2b/3b/8b/d20/d24). Standard pre-norm transformer, RMSNorm, SDPA attention, weight-tied head.
- `codechat/common.py` — `COMPUTE_DTYPE` (default `bfloat16`, override via env `CODECHAT_DTYPE`), `seed_all`, `DEVICE`.
- `codechat/dataloader.py` — `PretrainLoader` streams random `block_size+1` windows from `*.bin` uint16 shards. Each rank gets its own RNG seed (`seed + rank*9973`) so 8 GPUs see disjoint windows. `SFTLoader` reads `{input_ids, labels}` jsonl with `-100` masking on non-supervised tokens.
- `codechat/checkpoint.py` — single `save()` handles plain / DDP / FSDP transparently (the FSDP branch is what makes ckpts portable downstream).
- `codechat/optim.py` — AdamW only, decay/no-decay split by `p.ndim`. Cosine LR with warmup.
- `codechat/execution.py` — subprocess code-execution reward for MBPP RL. Two modes: `fractional` (binary-ish, k/n) and `tiered` (the staircase that rescued 8B RL — see below). Not a security sandbox; trusted hosts only.
- `codechat/funcall_reward.py` — dense staircase reward for function-calling RL: `no tag (0.00) → bad json (0.15) → no name (0.30) → wrong name (0.35) → name only (0.55) → partial args (0.55–0.99) → full match (1.00)`. The whole point is non-zero gradient signal from step 1 (see `docs/mixed_sft_vs_moe.md` and the v5/v6 reports for why the binary MBPP reward failed on 8B base).
- `codechat/tokenizer.py` — GPT-2 BPE via tiktoken (`VOCAB_SIZE=50257`). Chat tags `<|user|>`, `<|assistant|>`, `<|end|>` are plain text that gets BPE'd; no real special-token additions to vocab. Funcall pipelines additionally use `<|system|>`, `<|function_response|>`, and an inline `<functioncall>` sentinel (see `funcall_reward._extract_functioncall_json`).

Scripts pair up: each `prepare_*.py` builds a jsonl/bin for the matching trainer.

- `prepare_pretrain.py` → tokenizes `codeparrot/github-code-clean` Python subset → `data/pretrain/*.bin`
- `prepare_sft.py` / `prepare_sft_code.py` / `prepare_sft_funcall.py` → `data/sft*/train.jsonl`
- `prepare_rl_funcall.py` → extracts `(prompt, gt_name, gt_args)` triples from glaive-v2 → `data/rl_funcall/{train,eval}.jsonl`
- `base_train.py` / `chat_sft.py` / `chat_rl.py` / `chat_rl_funcall.py` — the four trainers; all share the FSDP-detect/wrap idiom and the same `setup_distributed` + `wrap_fsdp` helpers (duplicated, not factored out).
- `chat_cli.py` (free-form chat) and `funcall_cli.py` (tool-call REPL with optional auto-execution via `--executors my_tools.py`) — inference entry points.
- `eval_mbpp_pass_at_k.py` prints a `VERDICT:` line that gates whether GRPO is worth running: `pass@1 < 1%` → don't bother, `[1%, 5%)` → tiered + group≥8 + filter, `≥5%` → standard GRPO works.

## Pipeline versions and why they exist

The `train_a800_x8_v*` scripts are *successive attempts*, not alternatives — each one fixes the previous one's failure. The reports in `reports/` document what went wrong:

| Pipeline | Problem it solves | Output ckpt | Report |
|---|---|---|---|
| `train_a800_x8.sh` | Original 8B pretrain + SFT. **Stage-5 MBPP RL had reward ≡ 0** for 415 steps because pass@k ≈ 0 → group advantage ≡ 0 → no gradient. | `codechat_8b_sft` | `TRAINING_REPORT_8b_a88_x8.md` |
| `train_a800_x8_v2_funcall.sh` | Adds funcall SFT (glaive-v2) + "rescued" MBPP RL via `eval_mbpp_pass_at_k → filter_mbpp_by_passrate → tiered reward + group_size=8`. | `codechat_8b_sft_funcall` | `TRAINING_REPORT_8b_funcall_v2_v3.md` |
| `train_a800_x8_v3_funcall.sh` | Extra code SFT pass on top of `codechat_8b_sft` (MBPP non-train + Codeforces-Python + the-stack-smol) before funcall RL. | `codechat_8b_sft_code` | `docs/codechat_8b_sft_code.md` |
| `train_a800_x8_v5_funcall.sh` | Funcall RL on top of v3 base — first run with non-zero reward signal. | `codechat_8b_rl_funcall_v5` | `TRAINING_REPORT_8b_funcall_v5.md` |
| `train_a800_x8_v6.sh` | v5 became a funcall *specialist* — it emits `<functioncall>` even for "write quicksort". v6 fixes this with **joint** code+funcall SFT (shuffled mix), then funcall-only RL. Code ability comes from SFT, not RL, because MBPP RL still has zero signal on the 8B base. | `codechat_8b_sft_v6` / `codechat_8b_rl_v6` | `TRAINING_REPORT_8b_v6_unified.md` |

When asked to "improve the model" or "fix a regression", prefer the v6 path; v1/v2 MBPP RL is a known dead end on this base.

## Conventions worth knowing

- **Loss masking** in funcall SFT: only assistant-segment tokens get gradients; system / user / function_response tokens are `-100`. The model learns *when* to emit `<functioncall>` and how to phrase post-tool replies.
- **Run names matter** — they're the directory name under both `checkpoints/` and `runs/tb/`. Don't reuse a name across versions or you'll overwrite Tensorboard history.
- **Stage idempotence** — every shell pipeline has `SKIP_TO=N` and `FORCE_*=1` env knobs to rebuild specific artifacts without redoing the whole run. Prefer those over hand-editing the script.
- Repository documentation (including the 8B README) is partly in Chinese; mixed-language comments inside scripts are normal.
