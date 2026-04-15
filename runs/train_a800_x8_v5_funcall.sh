#!/usr/bin/env bash
# =============================================================================
# CodeChat 8B — v5 funcall pipeline, rebuilt around MathGPT's successful
# recipe. 8× A800-SXM4-80GB, bf16 + FSDP.
#
# What changed vs v2 / v3 (and why)
# ---------------------------------
#
# Prior versions tried to save MBPP RL with a tiered reward + pass-rate
# filter. That only works if the base can emit parseable Python at all,
# and our 8B SFT couldn't clear that bar (parseable ~1.75%, pass@8 = 0).
# Rather than keep patching the MBPP path we copy the MathGPT recipe that
# actually worked: SFT on exactly the distribution you'll RL on, then
# RL with a dense, format-based reward.
#
# Concretely:
#
#   1. **SFT and RL draw from the same dataset**. We SFT on
#      glaive-function-calling-v2 (teaching the <functioncall> JSON
#      format), then RL on held-out prompts from the *same* dataset
#      where ground-truth calls are known. This is the "GSM8K in SFT +
#      GSM8K in RL" trick MathGPT used — no distribution shift at RL
#      time, so reward starts non-zero and only has to go up.
#
#   2. **Reward is format-first, dense, staircase** (codechat/funcall_reward.py):
#        no <functioncall> tag              → 0.00
#        tag but JSON invalid               → 0.15
#        JSON missing name                  → 0.30
#        JSON parses, wrong function name   → 0.35
#        right name, args unparseable       → 0.55
#        right name + k/n arg match         → 0.55 + 0.45 * k/n
#        right name + all args match        → 1.00
#      Group variance is almost guaranteed, unlike MBPP binary pass/fail.
#
#   3. **Per-rank different prompts**. scripts/chat_rl_funcall.py assigns
#      each FSDP rank its own slice of the training problems — 8 ranks =
#      8 distinct prompts per step, versus the old code's "all ranks do
#      the same prompt via dist.broadcast". 8× the gradient diversity for
#      free.
#
#   4. **Batched rollouts**. num_samples=16 completions are sampled in
#      one forward pass (batch dim K), not a Python for-loop. Same
#      shape as MathGPT's engine.generate_batch.
#
#   5. **REINFORCE with baseline, no KL**. The dense reward + staircase
#      has enough self-regularization; removing the ref model halves
#      activation memory on 8B FSDP and buys us the num_samples budget.
#
#   6. **Online pass@k eval every 30 steps** on a 5% held-out split.
#      So we can pick best-Pass@1 or best-Pass@K afterwards instead of
#      blind-running to max_steps (MathGPT's best ckpt was at step 90
#      for Pass@K, not at the end).
#
#   7. **LR schedule**: init_lr_frac × base_lr, linearly decays to 0.
#      Same as MathGPT. Default init_lr_frac=0.05.
#
# Pipeline stages
# ---------------
#   [1] prepare funcall SFT jsonl     -> data/sft_funcall/train.jsonl
#   [2] funcall SFT (FSDP x8)         -> checkpoints/codechat_8b_funcall_v5/latest.pt
#   [3] extract RL prompts from glaive-> data/rl_funcall/{train,eval}.jsonl
#   [4] quick pre-RL diagnostic       -> print reward ladder on SFT ckpt
#   [5] funcall RL (FSDP x8)          -> checkpoints/codechat_8b_rl_funcall_v5/latest.pt
#
# Skip stages:
#   SKIP_TO=2  跳过数据准备，直接 SFT
#   SKIP_TO=3  已有 SFT ckpt，从抽 RL prompts 开始
#   SKIP_TO=5  已有 SFT ckpt + RL data，只跑 RL
#
# Prereqs:
#   - 8× A800 80GB
#   - checkpoints/codechat_8b_sft/latest.pt exists (from train_a800_x8.sh)
#   - HF可访问 glaiveai/glaive-function-calling-v2, or HF_HOME has it cached
# =============================================================================
set -euo pipefail

export CODECHAT_DTYPE=bfloat16
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN=${RUN:-codechat_8b}
FUNCALL_SFT_RUN=${FUNCALL_SFT_RUN:-${RUN}_funcall_v5}
RL_RUN=${RL_RUN:-${RUN}_rl_funcall_v5}

# Starting checkpoint for the v5 funcall SFT. Default to the general SFT
# ckpt produced by train_a800_x8.sh. Override to resume from a v2/v3
# funcall ckpt if you already trained one.
BASE_CKPT=${BASE_CKPT:-checkpoints/${RUN}_sft/latest.pt}

SFT_DATA_DIR=${SFT_DATA_DIR:-data/sft_funcall}
RL_DATA_DIR=${RL_DATA_DIR:-data/rl_funcall}

SFT_MAX_EXAMPLES=${SFT_MAX_EXAMPLES:-0}          # 0 = all ~113k rows
RL_MAX_EXAMPLES=${RL_MAX_EXAMPLES:-0}            # 0 = all rows with a call
RL_EVAL_RATIO=${RL_EVAL_RATIO:-0.05}

# SFT hyperparameters — longer than v2 because v5 puts MORE weight on
# getting SFT right (MathGPT's key lesson: 8× more SFT steps → 5.9× RL
# initial reward). 6000 steps × 131k tokens ≈ 786M supervised tokens.
SFT_LR=${SFT_LR:-3e-5}
SFT_MAX_STEPS=${SFT_MAX_STEPS:-6000}
SFT_WARMUP=${SFT_WARMUP:-200}

# RL hyperparameters. Mirrors MathGPT v2 (the run that worked):
#   num-samples=16-32, init-lr-frac=0.02-0.05, eval-every=30,
#   num-epochs=1 (their 3-epoch run overfit).
RL_NUM_EPOCHS=${RL_NUM_EPOCHS:-1}
RL_MAX_STEPS=${RL_MAX_STEPS:-0}                  # 0 = derive from num_epochs
RL_NUM_SAMPLES=${RL_NUM_SAMPLES:-16}
RL_MAX_NEW=${RL_MAX_NEW:-256}
RL_TEMPERATURE=${RL_TEMPERATURE:-1.0}
RL_TOP_K=${RL_TOP_K:-50}
RL_LR=${RL_LR:-1e-5}
RL_INIT_LR_FRAC=${RL_INIT_LR_FRAC:-0.05}
RL_EVAL_EVERY=${RL_EVAL_EVERY:-30}
RL_EVAL_EXAMPLES=${RL_EVAL_EXAMPLES:-200}
RL_SAVE_EVERY=${RL_SAVE_EVERY:-30}
RL_KEEP_EVERY=${RL_KEEP_EVERY:-60}
RL_LOG_ROLLOUTS_EVERY=${RL_LOG_ROLLOUTS_EVERY:-50}

NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29505}                # distinct from v1/v2/v3

SKIP_TO=${SKIP_TO:-1}

# ---------------------------------------------------------------------------
# Bootstrap training venv (same idiom as previous scripts)
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv_train"

if [ -z "${SYS_PYTHON:-}" ]; then
    for cand in python python3 python3.12 python3.11 python3.10; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import torch" >/dev/null 2>&1; then
            SYS_PYTHON="$(command -v "$cand")"
            break
        fi
    done
fi
if [ -z "${SYS_PYTHON:-}" ]; then
    echo "ERROR: no system python with torch. Set SYS_PYTHON=/path/to/python." >&2
    exit 1
fi
echo "==> [env] system python = ${SYS_PYTHON}"
"${SYS_PYTHON}" -c "import torch, sys; print(f'         -> python {sys.version.split()[0]}, torch {torch.__version__}, cuda {torch.version.cuda}')"

SYS_TORCH_SITE="$("${SYS_PYTHON}" -c 'import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))')"

if [ ! -d "${VENV_DIR}" ]; then
    echo "==> [env] creating training venv at ${VENV_DIR}"
    "${SYS_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --upgrade pip wheel
fi

PY="${VENV_DIR}/bin/python"

if ! "${PY}" -c "import torch" >/dev/null 2>&1; then
    echo "==> [env] venv can't import torch; appending host torch site via .pth"
    VENV_SITE="$("${PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    echo "${SYS_TORCH_SITE}" > "${VENV_SITE}/system_torch.pth"
fi

"${VENV_DIR}/bin/pip" install \
    "tensorboard>=2.16.0" \
    "tiktoken>=0.7.0" \
    "numpy>=1.26" \
    "tqdm>=4.66.0" \
    "datasets<4.0" \
    "huggingface_hub<0.24" \
    "fsspec<=2024.5.0" \
    "pyarrow" \
    "requests" \
    "httpx[socks]"

"${PY}" - <<'PYEOF'
import sys, torch, tensorboard, tiktoken, numpy, datasets
print(f"[env] python={sys.executable}")
print(f"[env] torch={torch.__version__}  cuda={torch.version.cuda}  n_gpu={torch.cuda.device_count()}")
print(f"[env] datasets={datasets.__version__}  tiktoken={tiktoken.__version__}")
PYEOF

TORCHRUN_CMD=("${PY}" -m torch.distributed.run)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
if [ ! -f "${BASE_CKPT}" ]; then
    echo "ERROR: base checkpoint missing: ${BASE_CKPT}" >&2
    echo "       Run runs/train_a800_x8.sh SFT stage first, or set BASE_CKPT." >&2
    exit 1
fi
echo "==> [check] base ckpt = ${BASE_CKPT} ($(du -h "${BASE_CKPT}" | cut -f1))"

# ===========================================================================
# Stage 1: funcall SFT data
# ===========================================================================
if [ "$SKIP_TO" -le 1 ]; then
    echo "==> [1/5] preparing funcall SFT data"
    if [ -s "${SFT_DATA_DIR}/train.jsonl" ] && [ "${FORCE_PREP_SFT:-0}" != "1" ]; then
        echo "    ${SFT_DATA_DIR}/train.jsonl exists ($(wc -l < "${SFT_DATA_DIR}/train.jsonl") lines), skipping."
        echo "    regenerate: FORCE_PREP_SFT=1 bash runs/train_a800_x8_v5_funcall.sh"
    else
        "${PY}" -m scripts.prepare_sft_funcall \
            --out-dir "${SFT_DATA_DIR}" \
            --max-examples "${SFT_MAX_EXAMPLES}"
    fi
fi

# ===========================================================================
# Stage 2: funcall SFT, MathGPT-style longer schedule
#
# Budget: device_batch=1 × grad_accum=8 × world_size=8 × seq=2048
#         ≈ 131k tokens/step × 6000 steps = 786M supervised tokens.
# That's ~7 epochs over 113k glaive rows — same per-example over-sampling
# principle as MathGPT's "GSM8K × 16 epochs in SFT".
# ===========================================================================
if [ "$SKIP_TO" -le 2 ]; then
    echo "==> [2/5] funcall SFT (FSDP x${NPROC}, base=${BASE_CKPT} -> ${FUNCALL_SFT_RUN})"
    "${TORCHRUN_CMD[@]}" \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m scripts.chat_sft \
            --base-ckpt "${BASE_CKPT}" \
            --data-dir "${SFT_DATA_DIR}" \
            --device-batch-size 1 \
            --grad-accum 8 \
            --lr "${SFT_LR}" \
            --warmup "${SFT_WARMUP}" \
            --max-steps "${SFT_MAX_STEPS}" \
            --run-name "${FUNCALL_SFT_RUN}"
fi

FUNCALL_SFT_CKPT="checkpoints/${FUNCALL_SFT_RUN}/latest.pt"

# ===========================================================================
# Stage 3: extract RL prompts + ground-truth calls from glaive
# ===========================================================================
if [ "$SKIP_TO" -le 3 ]; then
    echo "==> [3/5] extracting RL prompts + ground truth from glaive"
    if [ -s "${RL_DATA_DIR}/train.jsonl" ] && [ -s "${RL_DATA_DIR}/eval.jsonl" ] \
       && [ "${FORCE_PREP_RL:-0}" != "1" ]; then
        echo "    ${RL_DATA_DIR}/{train,eval}.jsonl exist "\
             "(train=$(wc -l < "${RL_DATA_DIR}/train.jsonl") "\
             "eval=$(wc -l < "${RL_DATA_DIR}/eval.jsonl")), skipping."
        echo "    regenerate: FORCE_PREP_RL=1 bash runs/train_a800_x8_v5_funcall.sh"
    else
        "${PY}" -m scripts.prepare_rl_funcall \
            --out-dir "${RL_DATA_DIR}" \
            --max-examples "${RL_MAX_EXAMPLES}" \
            --eval-ratio "${RL_EVAL_RATIO}"
    fi
fi

# ===========================================================================
# Stage 4: quick pre-RL diagnostic — print reward tier distribution on the
# funcall SFT ckpt over a handful of eval examples. Catches "the SFT can't
# even emit <functioncall>" failure mode before burning GPU hours.
# ===========================================================================
if [ "$SKIP_TO" -le 4 ]; then
    if [ ! -f "${FUNCALL_SFT_CKPT}" ]; then
        echo "ERROR: funcall SFT ckpt missing: ${FUNCALL_SFT_CKPT}" >&2
        exit 1
    fi
    if [ ! -s "${RL_DATA_DIR}/eval.jsonl" ]; then
        echo "WARN: ${RL_DATA_DIR}/eval.jsonl missing, skipping diagnostic"
    else
        echo "==> [4/5] pre-RL reward-tier diagnostic (FSDP x${NPROC})"
        # One eval pass with no gradient update. chat_rl_funcall auto-runs
        # an eval at step=1 when --eval-every > 0, so we just run it with
        # max-steps=1 and very small num_samples to get a cheap snapshot.
        "${TORCHRUN_CMD[@]}" \
            --standalone \
            --nproc_per_node="$NPROC" \
            --master_addr="$MASTER_ADDR" \
            --master_port="$MASTER_PORT" \
            -m scripts.chat_rl_funcall \
                --sft-ckpt "${FUNCALL_SFT_CKPT}" \
                --problems-file "${RL_DATA_DIR}/train.jsonl" \
                --eval-file "${RL_DATA_DIR}/eval.jsonl" \
                --run-name "${RL_RUN}_diag" \
                --max-steps 1 \
                --num-samples 8 \
                --eval-every 1 \
                --eval-examples 50 \
                --save-every 100000 \
                --lr 0 \
                --init-lr-frac 0
    fi
fi

# ===========================================================================
# Stage 5: funcall RL
# ===========================================================================
if [ "$SKIP_TO" -le 5 ]; then
    if [ ! -f "${FUNCALL_SFT_CKPT}" ]; then
        echo "ERROR: funcall SFT ckpt missing: ${FUNCALL_SFT_CKPT}" >&2
        exit 1
    fi
    if [ ! -s "${RL_DATA_DIR}/train.jsonl" ]; then
        echo "ERROR: RL train data missing: ${RL_DATA_DIR}/train.jsonl" >&2
        exit 1
    fi
    echo "==> [5/5] funcall RL (FSDP x${NPROC}, base=${FUNCALL_SFT_CKPT} -> ${RL_RUN})"
    echo "    train=$(wc -l < "${RL_DATA_DIR}/train.jsonl")  "\
         "eval=$(wc -l < "${RL_DATA_DIR}/eval.jsonl")  "\
         "num_samples=${RL_NUM_SAMPLES}  num_epochs=${RL_NUM_EPOCHS}  "\
         "init_lr_frac=${RL_INIT_LR_FRAC}"

    EXTRA_MAX_STEPS_FLAG=""
    if [ "${RL_MAX_STEPS}" != "0" ]; then
        EXTRA_MAX_STEPS_FLAG="--max-steps ${RL_MAX_STEPS}"
    fi

    "${TORCHRUN_CMD[@]}" \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m scripts.chat_rl_funcall \
            --sft-ckpt "${FUNCALL_SFT_CKPT}" \
            --problems-file "${RL_DATA_DIR}/train.jsonl" \
            --eval-file "${RL_DATA_DIR}/eval.jsonl" \
            --run-name "${RL_RUN}" \
            --num-epochs "${RL_NUM_EPOCHS}" \
            ${EXTRA_MAX_STEPS_FLAG} \
            --num-samples "${RL_NUM_SAMPLES}" \
            --max-new-tokens "${RL_MAX_NEW}" \
            --temperature "${RL_TEMPERATURE}" \
            --top-k "${RL_TOP_K}" \
            --lr "${RL_LR}" \
            --init-lr-frac "${RL_INIT_LR_FRAC}" \
            --eval-every "${RL_EVAL_EVERY}" \
            --eval-examples "${RL_EVAL_EXAMPLES}" \
            --save-every "${RL_SAVE_EVERY}" \
            --keep-every "${RL_KEEP_EVERY}" \
            --log-rollouts-every "${RL_LOG_ROLLOUTS_EVERY}"
fi

# ===========================================================================
# Summary
# ===========================================================================
echo ""
echo "================================================================"
echo "  v5 funcall pipeline complete"
echo "================================================================"
echo ""
echo "Checkpoints:"
echo "  funcall SFT: ${FUNCALL_SFT_CKPT}"
echo "  funcall RL:  checkpoints/${RL_RUN}/latest.pt"
echo ""
echo "TensorBoard:"
echo "  ${VENV_DIR}/bin/tensorboard --logdir runs/tb"
echo "  (compare codechat_8b_rl (v1 dead) vs ${RL_RUN} (v5 live reward curve))"
echo ""
echo "Best checkpoint selection:"
echo "  Look at runs/tb/${RL_RUN} -> eval/pass@1 and eval/pass@${RL_NUM_SAMPLES}"
echo "  The MathGPT playbook: Pass@1 peaks later (good for deployment),"
echo "  Pass@K peaks early (good for sampling-based majority vote)."
echo ""
echo "Smoke test:"
echo "  ${PY} -m scripts.chat_cli --ckpt checkpoints/${RL_RUN}/latest.pt"
echo ""
echo "Re-run partial stages:"
echo "  SKIP_TO=2 bash runs/train_a800_x8_v5_funcall.sh     # already prepared SFT data"
echo "  SKIP_TO=3 bash runs/train_a800_x8_v5_funcall.sh     # already have funcall SFT ckpt"
echo "  SKIP_TO=5 bash runs/train_a800_x8_v5_funcall.sh     # data + SFT ready, just RL"
echo "  FORCE_PREP_SFT=1 ...                                 # rebuild SFT jsonl"
echo "  FORCE_PREP_RL=1  ...                                 # rebuild RL jsonl"
