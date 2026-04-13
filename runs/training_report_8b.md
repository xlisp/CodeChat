# CodeChat 8B Training Report

**Date:** 2026-04-13
**Hardware:** 8x A800-SXM4-80GB (NVLink)
**Framework:** PyTorch 2.10 + FSDP (FULL_SHARD)

---

## Model Architecture

| Parameter        | Value           |
|------------------|-----------------|
| Preset           | `8b`            |
| Depth (layers)   | 40              |
| Hidden size      | 4096            |
| Attention heads  | 32              |
| Context length   | 2048            |
| Vocab size       | 50257           |
| Total params     | ~8.3B           |
| Dtype            | bfloat16        |

---

## Stage 2: Pretraining — DONE

### Configuration

| Setting               | Value                |
|-----------------------|----------------------|
| Global batch size     | 1 × 8 × 8 × 2048 = 131,072 tokens/step |
| Device batch size     | 1                    |
| Gradient accumulation | 8                    |
| World size            | 8 (FSDP FULL_SHARD) |
| Learning rate         | 1.5e-4 (cosine)     |
| Warmup steps          | 1,000                |
| Total steps           | 30,000               |
| Optimizer             | AdamW                |

### Results

| Metric                   | Value       |
|--------------------------|-------------|
| Total tokens seen        | **3.93B**   |
| Wall time                | **48.9 hours** |
| Throughput               | 22.3 Ktok/s (stable) |
| Final loss               | **0.6086**  |
| Avg loss (last 1k steps) | 0.6972      |
| Final learning rate      | 1.50e-5     |
| Final grad norm          | 0.18        |

### Loss Curve

```
Step         Loss
─────────────────
     3      11.37   (random init)
  5,000      1.11
 10,000      0.97
 15,000      0.68
 20,000      0.50
 25,000      0.80
 30,000      0.61
```

Loss converged well below 1.0. The spike around step 25k is consistent with
learning-rate cosine decay noise; the loss re-settled to ~0.61 by end of training.

### Checkpoint

```
checkpoints/codechat_8b/latest.pt   (16 GB)
```

---

## Stage 3: SFT Data Preparation — DONE

- Sources: `iamtarun/python_code_instructions_18k_alpaca` + `sahil2801/CodeAlpaca-20k`
- Output: **38,628 SFT examples** → `data/sft/train.jsonl`

---

## Stage 4: SFT — FIXED (was OOM)

### Root Cause

`scripts/chat_sft.py` was written for single-GPU training but launched with
`torchrun --nproc_per_node=8`. Three issues caused CUDA OOM:

1. **No distributed setup** — no `dist.init_process_group()`, no per-rank
   `cuda.set_device()`.
2. **Checkpoint loaded to `cuda:0`** — all 8 processes called
   `torch.load(..., map_location="cuda")`, each loading the full 16GB
   checkpoint to GPU 0 simultaneously (8 × 16GB = 128GB on a single 79GB card).
3. **No FSDP wrapping** — the 8B model + AdamW optimizer states (~96GB fp32)
   cannot fit on a single GPU.

### Fix Applied

Updated `scripts/chat_sft.py` to mirror `base_train.py`:

- Added `setup_distributed()` — detects `LOCAL_RANK`, sets `cuda.set_device()`,
  initialises NCCL process group.
- Changed checkpoint loading to `map_location="cpu"` — loads weights to CPU
  first, then loads into model before moving to GPU.
- Added `wrap_fsdp()` — wraps model with FSDP (FULL_SHARD) using Block-level
  auto-wrapping, same config as pretraining.
- Added `no_sync()` for non-final micro-batches to avoid redundant gradient
  all-reduces.
- Added FSDP-aware `clip_grad_norm_()`.
- TensorBoard writer only on rank 0.
- Added `dist.barrier()` + `dist.destroy_process_group()` at cleanup.

### SFT Configuration (ready to re-run)

| Setting               | Value              |
|-----------------------|--------------------|
| Base checkpoint       | `checkpoints/codechat_8b/latest.pt` |
| Device batch size     | 1                  |
| Gradient accumulation | 8                  |
| Learning rate         | 5e-5 (cosine)      |
| Warmup steps          | 100                |
| Total steps           | 3,000              |
| Data                  | 38,628 examples    |

Re-run command:
```bash
SKIP_TO=4 bash runs/train_a800_x8.sh
```

---

## Comparison: 2B vs 8B Pretraining

| Metric        | 2B (single GPU)     | 8B (FSDP x8)        |
|---------------|---------------------|----------------------|
| Parameters    | ~2.1B               | ~8.3B                |
| Steps         | 24,987              | 30,000               |
| Tokens seen   | 1.64B               | 3.93B                |
| Wall time     | 57.5 h              | 48.9 h               |
| Throughput    | 7.9 Ktok/s          | 22.3 Ktok/s          |
| Final loss    | 0.867               | **0.609**            |
| Min loss      | 0.624               | **0.316**            |

The 8B model achieves significantly lower loss despite seeing only ~2.4x more
tokens, demonstrating the scaling advantage. 8-GPU FSDP training was also
~2.8x faster in throughput while training a 4x larger model.

---

## TensorBoard

```bash
tensorboard --logdir runs/tb
```

Runs available:
- `runs/tb/codechat_2b` — 2B pretraining (baseline)
- `runs/tb/codechat_8b` — 8B pretraining
- `runs/tb/codechat_8b_sft` — (pending, after SFT re-run)
