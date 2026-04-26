#!/usr/bin/env bash
# =============================================================================
# CodeChat 8B — v7 unified pipeline.
# 8× A800-SXM4-80GB, bf16 + FSDP.
#
# Why v7
# ------
# v6 produced a single ckpt that did funcall well (pass@1 86%) but completely
# lost code ability — "write quicksort" came back as token soup. Root cause
# was the data mix: 113k funcall rows × ~7.5k tokens vs 20k code rows × ~1.5k
# tokens = ~30:1 supervised-token imbalance. v6 also lacked any per-domain
# verification during the 14h SFT, so the regression wasn't caught until the
# final smoke test.
#
# v7 fixes this with three changes (rationale: see reports/TRAINING_REPORT_8b_v6_unified.md):
#
#   1. UNIFIED DATA FORMAT (minimind-style conversations, see
#      scripts/prepare_sft_v7.py). One file, both task families. Token-balanced
#      to ~1:1 by capping glaive at 30k rows and expanding code sources
#      (MBPP non-train + Codeforces + the-stack-smol + HumanEval + CodeAlpaca).
#
#   2. DISCRIMINATIVE NEGATIVES + 20% SYSTEM INJECTION
#      v6 only ever showed <|system|> together with funcall data, so the model
#      learned "system tag → emit <functioncall>". v7 fixes both directions:
#        - codechat/dataloader.py SFTConvLoader: 20% chance to prepend a
#          no-tools system prompt to code samples (so <|system|> appears in
#          plain-chat training too)
#        - prepare_sft_v7.py synthesizes ~10% negative samples: real funcall
#          system block + code Q + code A (no <functioncall>) — teaches that
#          tools-in-system ≠ must-call-tool
#
#   3. PER-DOMAIN EVAL DURING SFT (scripts/chat_sft_v7.py)
#      Every --eval-every steps: held-out loss separately on code and funcall.
#      Every --smoke-every steps: greedy quicksort + weather generation, write
#      sft/smoke_code_pass and sft/smoke_funcall_pass to TB. Catches v6-style
#      regression in real time, not 14h after the fact.
#
# RL: default SKIP. v5 and v6 both showed funcall RL adds <1% pass@1 from an
# 85% SFT starting point (REINFORCE has no advantage signal when reward
# variance ≈ 0). Set RUN_RL=1 to opt in for a short 60-step run.
#
# Pipeline stages
# ---------------
#   [1] prepare unified SFT data  -> data/sft_v7/{train,eval_code,eval_funcall}.jsonl
#   [2] joint SFT (FSDP x8)       -> checkpoints/codechat_8b_sft_v7/latest.pt
#                                    + per-domain eval + smoke every N steps
#   [3] dual smoke test (chat_cli + funcall_cli) on the SFT ckpt
#   [4] (RUN_RL=1 only) extract RL prompts (reuse v5 data if present)
#   [5] (RUN_RL=1 only) short funcall RL — 60 steps, mirrors v5 recipe
#
# Skip stages:
#   SKIP_TO=2  data ready, jump to SFT
#   SKIP_TO=3  SFT ckpt ready, just smoke
#   SKIP_TO=4  request RL (also requires RUN_RL=1)
#
# Force regen:
#   FORCE_PREP=1  rebuild data/sft_v7/* even if it exists
#
# Prereqs:
#   - 8× A800 80GB
#   - checkpoints/codechat_8b_sft/latest.pt exists (neutral SFT base, NOT v6)
#   - HF: glaive-function-calling-v2, MBPP, Codeforces-Python-Submissions,
#         the-stack-smol, openai_humaneval, sahil2801/CodeAlpaca-20k
# =============================================================================
set -euo pipefail

export CODECHAT_DTYPE=bfloat16
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN=${RUN:-codechat_8b}
SFT_V7_RUN=${SFT_V7_RUN:-${RUN}_sft_v7}
RL_RUN=${RL_RUN:-${RUN}_rl_v7}

# v7 starts from the neutral SFT base, NOT from v6 ckpt (which is funcall-skewed).
BASE_CKPT=${BASE_CKPT:-checkpoints/${RUN}_sft/latest.pt}

SFT_V7_DIR=${SFT_V7_DIR:-data/sft_v7}
RL_DATA_DIR=${RL_DATA_DIR:-data/rl_funcall}

# Data prep knobs (passed through to prepare_sft_v7.py)
PREP_FUNCALL_CAP=${PREP_FUNCALL_CAP:-30000}
PREP_NEG_FRAC=${PREP_NEG_FRAC:-0.10}
PREP_MAX_CODEFORCES=${PREP_MAX_CODEFORCES:-60000}
PREP_MAX_THE_STACK=${PREP_MAX_THE_STACK:-60000}
PREP_MAX_CODEALPACA=${PREP_MAX_CODEALPACA:-20000}

# SFT hyperparameters — 6000 steps × 131k tokens/step ≈ 786M supervised tokens.
# v6 went to 8000 and started overfitting funcall around step 3000 anyway; we
# trim to 6000 because the token-balanced corpus has less duplicated funcall
# pressure to absorb.
SFT_LR=${SFT_LR:-3e-5}
SFT_MAX_STEPS=${SFT_MAX_STEPS:-6000}
SFT_WARMUP=${SFT_WARMUP:-300}
SFT_EVAL_EVERY=${SFT_EVAL_EVERY:-200}
SFT_SMOKE_EVERY=${SFT_SMOKE_EVERY:-500}
SFT_SAVE_EVERY=${SFT_SAVE_EVERY:-500}
SFT_SYSTEM_INJECT_RATIO=${SFT_SYSTEM_INJECT_RATIO:-0.20}

# RL knobs (only used if RUN_RL=1)
RUN_RL=${RUN_RL:-0}
RL_NUM_SAMPLES=${RL_NUM_SAMPLES:-16}
RL_MAX_STEPS=${RL_MAX_STEPS:-60}        # v5 plateaued at step 60; don't waste compute
RL_LR=${RL_LR:-1e-5}
RL_INIT_LR_FRAC=${RL_INIT_LR_FRAC:-0.05}
RL_EVAL_EVERY=${RL_EVAL_EVERY:-20}
RL_EVAL_EXAMPLES=${RL_EVAL_EXAMPLES:-120}
RL_SAVE_EVERY=${RL_SAVE_EVERY:-30}
RL_MAX_NEW=${RL_MAX_NEW:-256}

NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29507}        # distinct from v1/v2/v3/v5/v6

SKIP_TO=${SKIP_TO:-1}

# ---------------------------------------------------------------------------
# Bootstrap training venv (identical idiom to v5/v6)
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
# Stage 1: prepare unified SFT data
# ===========================================================================
if [ "$SKIP_TO" -le 1 ]; then
    echo "==> [1/5] preparing v7 unified SFT data"
    if [ -s "${SFT_V7_DIR}/train.jsonl" ] \
       && [ -s "${SFT_V7_DIR}/eval_code.jsonl" ] \
       && [ -s "${SFT_V7_DIR}/eval_funcall.jsonl" ] \
       && [ "${FORCE_PREP:-0}" != "1" ]; then
        echo "    ${SFT_V7_DIR}/{train,eval_code,eval_funcall}.jsonl exist:"
        echo "      train       = $(wc -l < "${SFT_V7_DIR}/train.jsonl") rows"
        echo "      eval_code   = $(wc -l < "${SFT_V7_DIR}/eval_code.jsonl") rows"
        echo "      eval_funcall= $(wc -l < "${SFT_V7_DIR}/eval_funcall.jsonl") rows"
        echo "    skipping (FORCE_PREP=1 to rebuild)"
    else
        "${PY}" -m scripts.prepare_sft_v7 \
            --out-dir "${SFT_V7_DIR}" \
            --max-codeforces "${PREP_MAX_CODEFORCES}" \
            --max-the-stack "${PREP_MAX_THE_STACK}" \
            --max-codealpaca "${PREP_MAX_CODEALPACA}" \
            --funcall-cap "${PREP_FUNCALL_CAP}" \
            --negative-frac "${PREP_NEG_FRAC}"
    fi
fi

# ===========================================================================
# Stage 2: joint SFT with per-domain eval + smoke
# ===========================================================================
if [ "$SKIP_TO" -le 2 ]; then
    echo "==> [2/5] joint SFT (FSDP x${NPROC}, base=${BASE_CKPT} -> ${SFT_V7_RUN})"
    echo "    eval_every=${SFT_EVAL_EVERY}  smoke_every=${SFT_SMOKE_EVERY}  "\
         "save_every=${SFT_SAVE_EVERY}  inject_ratio=${SFT_SYSTEM_INJECT_RATIO}"
    "${TORCHRUN_CMD[@]}" \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m scripts.chat_sft_v7 \
            --base-ckpt "${BASE_CKPT}" \
            --data-dir "${SFT_V7_DIR}" \
            --device-batch-size 1 \
            --grad-accum 8 \
            --lr "${SFT_LR}" \
            --warmup "${SFT_WARMUP}" \
            --max-steps "${SFT_MAX_STEPS}" \
            --eval-every "${SFT_EVAL_EVERY}" \
            --smoke-every "${SFT_SMOKE_EVERY}" \
            --save-every "${SFT_SAVE_EVERY}" \
            --system-inject-ratio "${SFT_SYSTEM_INJECT_RATIO}" \
            --run-name "${SFT_V7_RUN}"
fi

SFT_V7_CKPT="checkpoints/${SFT_V7_RUN}/latest.pt"

# ===========================================================================
# Stage 3: dual smoke test on the SFT ckpt
# ===========================================================================
if [ "$SKIP_TO" -le 3 ]; then
    if [ ! -f "${SFT_V7_CKPT}" ]; then
        echo "ERROR: ${SFT_V7_CKPT} missing; cannot run smoke" >&2
        exit 1
    fi
    echo ""
    echo "==> [3/5] dual smoke test on ${SFT_V7_CKPT}"
    echo ""
    echo "--- code task: quicksort ---"
    "${PY}" -m scripts.chat_cli \
        --ckpt "${SFT_V7_CKPT}" \
        --user "Write a Python implementation of quicksort." \
        --max-new-tokens 400 || echo "(chat_cli failed, continuing)"

    echo ""
    echo "--- funcall task: weather ---"
    "${PY}" -m scripts.funcall_cli \
        --ckpt "${SFT_V7_CKPT}" \
        --user "Weather in Tokyo?" || echo "(funcall_cli failed, continuing)"
fi

# ===========================================================================
# Stage 4: (opt-in) extract RL prompts — only if RUN_RL=1
# ===========================================================================
if [ "$SKIP_TO" -le 4 ] && [ "${RUN_RL}" = "1" ]; then
    echo "==> [4/5] extracting RL prompts (RUN_RL=1)"
    if [ -s "${RL_DATA_DIR}/train.jsonl" ] && [ -s "${RL_DATA_DIR}/eval.jsonl" ] \
       && [ "${FORCE_PREP_RL:-0}" != "1" ]; then
        echo "    ${RL_DATA_DIR}/{train,eval}.jsonl exist "\
             "(train=$(wc -l < "${RL_DATA_DIR}/train.jsonl") "\
             "eval=$(wc -l < "${RL_DATA_DIR}/eval.jsonl")), skipping."
    else
        "${PY}" -m scripts.prepare_rl_funcall \
            --out-dir "${RL_DATA_DIR}" \
            --max-examples 0 \
            --eval-ratio 0.05
    fi
fi

# ===========================================================================
# Stage 5: (opt-in) short funcall RL — only if RUN_RL=1
# ===========================================================================
if [ "$SKIP_TO" -le 5 ] && [ "${RUN_RL}" = "1" ]; then
    if [ ! -f "${SFT_V7_CKPT}" ]; then
        echo "ERROR: v7 SFT ckpt missing: ${SFT_V7_CKPT}" >&2
        exit 1
    fi
    echo "==> [5/5] funcall RL (RUN_RL=1, base=${SFT_V7_CKPT} -> ${RL_RUN})"
    echo "    max_steps=${RL_MAX_STEPS} num_samples=${RL_NUM_SAMPLES} "\
         "init_lr_frac=${RL_INIT_LR_FRAC}"
    "${TORCHRUN_CMD[@]}" \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m scripts.chat_rl_funcall \
            --sft-ckpt "${SFT_V7_CKPT}" \
            --problems-file "${RL_DATA_DIR}/train.jsonl" \
            --eval-file "${RL_DATA_DIR}/eval.jsonl" \
            --run-name "${RL_RUN}" \
            --max-steps "${RL_MAX_STEPS}" \
            --num-samples "${RL_NUM_SAMPLES}" \
            --max-new-tokens "${RL_MAX_NEW}" \
            --lr "${RL_LR}" \
            --init-lr-frac "${RL_INIT_LR_FRAC}" \
            --eval-every "${RL_EVAL_EVERY}" \
            --eval-examples "${RL_EVAL_EXAMPLES}" \
            --save-every "${RL_SAVE_EVERY}"
fi

RL_V7_CKPT="checkpoints/${RL_RUN}/latest.pt"

# ===========================================================================
# Summary
# ===========================================================================
echo ""
echo "================================================================"
echo "  v7 unified pipeline complete"
echo "================================================================"
echo ""
echo "Checkpoints:"
echo "  joint SFT: ${SFT_V7_CKPT}"
if [ "${RUN_RL}" = "1" ]; then
    echo "  funcall RL: ${RL_V7_CKPT}"
else
    echo "  (RL skipped; pass RUN_RL=1 to enable a 60-step funcall RL stage)"
fi
echo ""
echo "TensorBoard:"
echo "  ${VENV_DIR}/bin/tensorboard --logdir runs/tb"
echo "  Watch: sft/loss_code vs sft/loss_funcall (per-domain regression alarm)"
echo "         sft/smoke_code_pass + sft/smoke_funcall_pass (binary pass/fail)"
echo ""
echo "Expected behavior at convergence:"
echo "  - sft/loss_code and sft/loss_funcall both decrease and stabilize"
echo "  - smoke_code_pass and smoke_funcall_pass both reach 1 by ~step 2000"
echo "  - If smoke_code_pass drops to 0 mid-training, code is being overwritten"
echo "    -> kill the run, lower --funcall-cap or raise --negative-frac"
echo ""
echo "Manual tests:"
echo "  ${PY} -m scripts.chat_cli    --ckpt ${SFT_V7_CKPT} --user 'write quicksort'"
echo "  ${PY} -m scripts.funcall_cli --ckpt ${SFT_V7_CKPT} --user 'Weather in Tokyo?'"
echo ""
echo "Re-run partial stages:"
echo "  SKIP_TO=2 bash runs/train_a800_x8_v7.sh           # data ready, jump to SFT"
echo "  SKIP_TO=3 bash runs/train_a800_x8_v7.sh           # SFT ckpt ready, just smoke"
echo "  RUN_RL=1 SKIP_TO=4 bash runs/train_a800_x8_v7.sh  # opt in to RL stage"
echo "  FORCE_PREP=1 bash runs/train_a800_x8_v7.sh        # rebuild data/sft_v7/*"
