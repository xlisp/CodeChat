#!/usr/bin/env bash
# =============================================================================
# CodeChat 8B — v6 unified pipeline: code + funcall in one model.
# 8× A800-SXM4-80GB, bf16 + FSDP.
#
# Goal
# ----
# Produce a single ckpt that can BOTH write code (quicksort, MBPP-style
# problems) AND emit <functioncall> JSON when given a tool schema.
# v5 was a funcall specialist — it now emits <functioncall> for every
# prompt, so "write quicksort" comes back as a bogus tool call. v6 fixes
# this by training SFT jointly on code and funcall data.
#
# What changed vs v5
# ------------------
#
# 1. **Joint SFT on mixed distribution**. Concatenate sft_code/train.jsonl
#    and sft_funcall/train.jsonl, shuffle, train. The presence/absence of
#    a <|system|> tool block in the prompt acts as the disambiguator at
#    inference time — the model learns "system block present → emit
#    <functioncall>" and "no system block → write code directly".
#
# 2. **Start from codechat_8b_sft, not from v5 funcall ckpt**. v5's RL ckpt
#    is funcall-saturated: ~86% of its eval rollouts are `<functioncall>`
#    JSON. Fine-tuning it back toward code would fight that prior. Starting
#    from the neutral SFT base is cheaper and cleaner.
#
# 3. **Longer SFT schedule (8000 steps, up from 6000)**. Two distributions
#    share the budget, so give it 33% more steps to converge on both.
#
# 4. **Code-only RL is deliberately NOT run**. v1/v2 proved MBPP executable
#    reward produces zero gradient on the 8B base (pass@8 = 0 at RL start).
#    Code ability in v6 comes from SFT; only funcall gets RL because its
#    reward ladder (codechat/funcall_reward.py) has signal from step 1.
#
# 5. **Dual smoke test**. Final stage runs both chat_cli (quicksort check)
#    and funcall_cli (weather check) on the same ckpt.
#
# Pipeline stages
# ---------------
#   [1] prepare code SFT          -> data/sft_code/train.jsonl
#   [2] prepare funcall SFT       -> data/sft_funcall/train.jsonl
#   [3] merge + shuffle           -> data/sft_v6/train.jsonl
#   [4] joint SFT (FSDP x8)       -> checkpoints/codechat_8b_sft_v6/latest.pt
#   [5] extract RL prompts        -> data/rl_funcall/{train,eval}.jsonl (reused)
#   [6] pre-RL reward diagnostic  -> print tier distribution on v6 SFT ckpt
#   [7] funcall RL (FSDP x8)      -> checkpoints/codechat_8b_rl_v6/latest.pt
#   [8] dual smoke test           -> quicksort + weather tool
#
# Skip stages:
#   SKIP_TO=3  已经有 sft_code 和 sft_funcall，直接合并
#   SKIP_TO=4  合并好的 sft_v6 已存在，直接 SFT
#   SKIP_TO=5  已有 v6 SFT ckpt，从抽 RL prompts 开始
#   SKIP_TO=7  SFT + RL data 都 ready，只跑 RL
#   SKIP_TO=8  只跑 smoke test
#
# Prereqs:
#   - 8× A800 80GB
#   - checkpoints/codechat_8b_sft/latest.pt exists
#   - HF 可访问 glaiveai/glaive-function-calling-v2 + MBPP + Codeforces-Python
# =============================================================================
set -euo pipefail

export CODECHAT_DTYPE=bfloat16
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN=${RUN:-codechat_8b}
SFT_V6_RUN=${SFT_V6_RUN:-${RUN}_sft_v6}
RL_RUN=${RL_RUN:-${RUN}_rl_v6}

# Start from the neutral general SFT ckpt, NOT from v5 funcall (see rationale #2).
BASE_CKPT=${BASE_CKPT:-checkpoints/${RUN}_sft/latest.pt}

SFT_CODE_DIR=${SFT_CODE_DIR:-data/sft_code}
SFT_FUNCALL_DIR=${SFT_FUNCALL_DIR:-data/sft_funcall}
SFT_V6_DIR=${SFT_V6_DIR:-data/sft_v6}
RL_DATA_DIR=${RL_DATA_DIR:-data/rl_funcall}

SFT_FUNCALL_MAX=${SFT_FUNCALL_MAX:-0}             # 0 = all ~113k rows
SFT_CODE_MAX=${SFT_CODE_MAX:-0}                   # 0 = all rows
RL_MAX_EXAMPLES=${RL_MAX_EXAMPLES:-0}
RL_EVAL_RATIO=${RL_EVAL_RATIO:-0.05}

# SFT hyperparameters — 8000 steps × 131k tokens/step ≈ 1.05B supervised tokens
# across ~113k glaive rows + ~30k code rows (≈ 143k total, ~7 epochs).
SFT_LR=${SFT_LR:-3e-5}
SFT_MAX_STEPS=${SFT_MAX_STEPS:-8000}
SFT_WARMUP=${SFT_WARMUP:-300}

# RL hyperparameters — mirror v5 (the run that worked). Expect it to plateau
# around step 60-90 same as v5.
RL_NUM_EPOCHS=${RL_NUM_EPOCHS:-1}
RL_MAX_STEPS=${RL_MAX_STEPS:-0}
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
MASTER_PORT=${MASTER_PORT:-29506}                  # distinct from v1/v2/v3/v5

SKIP_TO=${SKIP_TO:-1}

# ---------------------------------------------------------------------------
# Bootstrap training venv (identical idiom to v5)
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
# Stage 1: code SFT data
# ===========================================================================
if [ "$SKIP_TO" -le 1 ]; then
    echo "==> [1/8] preparing code SFT data"
    if [ -s "${SFT_CODE_DIR}/train.jsonl" ] && [ "${FORCE_PREP_CODE:-0}" != "1" ]; then
        echo "    ${SFT_CODE_DIR}/train.jsonl exists ($(wc -l < "${SFT_CODE_DIR}/train.jsonl") lines), skipping."
        echo "    regenerate: FORCE_PREP_CODE=1 bash runs/train_a800_x8_v6.sh"
    else
        "${PY}" -m scripts.prepare_sft_code \
            --out-dir "${SFT_CODE_DIR}" \
            --max-examples "${SFT_CODE_MAX}"
    fi
fi

# ===========================================================================
# Stage 2: funcall SFT data
# ===========================================================================
if [ "$SKIP_TO" -le 2 ]; then
    echo "==> [2/8] preparing funcall SFT data"
    if [ -s "${SFT_FUNCALL_DIR}/train.jsonl" ] && [ "${FORCE_PREP_FUNCALL:-0}" != "1" ]; then
        echo "    ${SFT_FUNCALL_DIR}/train.jsonl exists ($(wc -l < "${SFT_FUNCALL_DIR}/train.jsonl") lines), skipping."
        echo "    regenerate: FORCE_PREP_FUNCALL=1 bash runs/train_a800_x8_v6.sh"
    else
        "${PY}" -m scripts.prepare_sft_funcall \
            --out-dir "${SFT_FUNCALL_DIR}" \
            --max-examples "${SFT_FUNCALL_MAX}"
    fi
fi

# ===========================================================================
# Stage 3: merge + shuffle into joint SFT jsonl
#
# Concat is fine: each line is a self-contained {input_ids, labels} pair,
# and the chat-tag disambiguator (<|system|> present/absent) is baked into
# the tokens — the loader doesn't need to know which source a row came from.
# shuf so the two distributions interleave across training steps, otherwise
# the first half of an epoch would be pure code and the second half pure
# funcall → huge gradient-distribution shift mid-epoch.
# ===========================================================================
if [ "$SKIP_TO" -le 3 ]; then
    echo "==> [3/8] merging code + funcall SFT data"
    if [ -s "${SFT_V6_DIR}/train.jsonl" ] && [ "${FORCE_MERGE:-0}" != "1" ]; then
        echo "    ${SFT_V6_DIR}/train.jsonl exists ($(wc -l < "${SFT_V6_DIR}/train.jsonl") lines), skipping."
        echo "    regenerate: FORCE_MERGE=1 bash runs/train_a800_x8_v6.sh"
    else
        mkdir -p "${SFT_V6_DIR}"
        N_CODE=$(wc -l < "${SFT_CODE_DIR}/train.jsonl")
        N_FUNC=$(wc -l < "${SFT_FUNCALL_DIR}/train.jsonl")
        cat "${SFT_CODE_DIR}/train.jsonl" "${SFT_FUNCALL_DIR}/train.jsonl" \
            | shuf --random-source=<(yes 42 2>/dev/null) \
            > "${SFT_V6_DIR}/train.jsonl"
        N_TOTAL=$(wc -l < "${SFT_V6_DIR}/train.jsonl")
        echo "    merged: code=${N_CODE}  funcall=${N_FUNC}  total=${N_TOTAL}"
        echo "    -> ${SFT_V6_DIR}/train.jsonl"
    fi
fi

# ===========================================================================
# Stage 4: joint SFT on merged data
#
# Budget: device_batch=1 × grad_accum=8 × world_size=8 × seq=2048
#         ≈ 131k tokens/step × 8000 steps = 1.05B supervised tokens
# At ~143k total rows, that's ~7 epochs. Same per-example pressure as v5's
# 786M/113k ≈ 7 epochs.
# ===========================================================================
if [ "$SKIP_TO" -le 4 ]; then
    echo "==> [4/8] joint SFT (FSDP x${NPROC}, base=${BASE_CKPT} -> ${SFT_V6_RUN})"
    "${TORCHRUN_CMD[@]}" \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m scripts.chat_sft \
            --base-ckpt "${BASE_CKPT}" \
            --data-dir "${SFT_V6_DIR}" \
            --device-batch-size 1 \
            --grad-accum 8 \
            --lr "${SFT_LR}" \
            --warmup "${SFT_WARMUP}" \
            --max-steps "${SFT_MAX_STEPS}" \
            --run-name "${SFT_V6_RUN}"
fi

SFT_V6_CKPT="checkpoints/${SFT_V6_RUN}/latest.pt"

# ===========================================================================
# Stage 5: extract RL prompts (reuse v5's data if present)
# ===========================================================================
if [ "$SKIP_TO" -le 5 ]; then
    echo "==> [5/8] extracting RL prompts (funcall only)"
    if [ -s "${RL_DATA_DIR}/train.jsonl" ] && [ -s "${RL_DATA_DIR}/eval.jsonl" ] \
       && [ "${FORCE_PREP_RL:-0}" != "1" ]; then
        echo "    ${RL_DATA_DIR}/{train,eval}.jsonl exist "\
             "(train=$(wc -l < "${RL_DATA_DIR}/train.jsonl") "\
             "eval=$(wc -l < "${RL_DATA_DIR}/eval.jsonl")), skipping."
    else
        "${PY}" -m scripts.prepare_rl_funcall \
            --out-dir "${RL_DATA_DIR}" \
            --max-examples "${RL_MAX_EXAMPLES}" \
            --eval-ratio "${RL_EVAL_RATIO}"
    fi
fi

# ===========================================================================
# Stage 6: pre-RL diagnostic on the joint SFT ckpt
#
# Critical check — v6's mixed SFT may have weakened funcall format fluency
# vs v5's pure-funcall SFT. If the tier distribution shows <50% full_match
# at step 1, funcall RL will still work (that was exactly v5's pre-RL
# starting point too), but we want to see the delta vs v5 here.
# ===========================================================================
if [ "$SKIP_TO" -le 6 ]; then
    if [ ! -f "${SFT_V6_CKPT}" ]; then
        echo "ERROR: v6 SFT ckpt missing: ${SFT_V6_CKPT}" >&2
        exit 1
    fi
    if [ ! -s "${RL_DATA_DIR}/eval.jsonl" ]; then
        echo "WARN: ${RL_DATA_DIR}/eval.jsonl missing, skipping diagnostic"
    else
        echo "==> [6/8] pre-RL reward-tier diagnostic (FSDP x${NPROC})"
        "${TORCHRUN_CMD[@]}" \
            --standalone \
            --nproc_per_node="$NPROC" \
            --master_addr="$MASTER_ADDR" \
            --master_port="$MASTER_PORT" \
            -m scripts.chat_rl_funcall \
                --sft-ckpt "${SFT_V6_CKPT}" \
                --problems-file "${RL_DATA_DIR}/train.jsonl" \
                --eval-file "${RL_DATA_DIR}/eval.jsonl" \
                --run-name "${RL_RUN}_diag" \
                --max-steps 1 \
                --num-samples 8 \
                --eval-every 1 \
                --eval-examples 48 \
                --save-every 100000 \
                --lr 0 \
                --init-lr-frac 0
    fi
fi

# ===========================================================================
# Stage 7: funcall RL — same recipe as v5
# ===========================================================================
if [ "$SKIP_TO" -le 7 ]; then
    if [ ! -f "${SFT_V6_CKPT}" ]; then
        echo "ERROR: v6 SFT ckpt missing: ${SFT_V6_CKPT}" >&2
        exit 1
    fi
    if [ ! -s "${RL_DATA_DIR}/train.jsonl" ]; then
        echo "ERROR: RL train data missing: ${RL_DATA_DIR}/train.jsonl" >&2
        exit 1
    fi
    echo "==> [7/8] funcall RL (FSDP x${NPROC}, base=${SFT_V6_CKPT} -> ${RL_RUN})"
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
            --sft-ckpt "${SFT_V6_CKPT}" \
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

RL_V6_CKPT="checkpoints/${RL_RUN}/latest.pt"

# ===========================================================================
# Stage 8: dual smoke test — verify the ckpt handles BOTH tasks
# ===========================================================================
if [ "$SKIP_TO" -le 8 ]; then
    if [ ! -f "${RL_V6_CKPT}" ]; then
        echo "WARN: ${RL_V6_CKPT} not found; running smoke test on SFT ckpt instead"
        SMOKE_CKPT="${SFT_V6_CKPT}"
    else
        SMOKE_CKPT="${RL_V6_CKPT}"
    fi
    echo ""
    echo "==> [8/8] dual smoke test on ${SMOKE_CKPT}"
    echo ""
    echo "--- code task: quicksort ---"
    "${PY}" -m scripts.chat_cli \
        --ckpt "${SMOKE_CKPT}" \
        --user "Write a Python implementation of quicksort." \
        --max-new-tokens 400 || echo "(chat_cli failed, continuing)"

    echo ""
    echo "--- funcall task: weather ---"
    "${PY}" -m scripts.funcall_cli \
        --ckpt "${SMOKE_CKPT}" \
        --user "Weather in Tokyo?" || echo "(funcall_cli failed, continuing)"
fi

# ===========================================================================
# Summary
# ===========================================================================
echo ""
echo "================================================================"
echo "  v6 unified pipeline complete"
echo "================================================================"
echo ""
echo "Checkpoints:"
echo "  joint SFT: ${SFT_V6_CKPT}"
echo "  final RL:  ${RL_V6_CKPT}"
echo ""
echo "TensorBoard:"
echo "  ${VENV_DIR}/bin/tensorboard --logdir runs/tb"
echo "  Compare: ${SFT_V6_RUN} (SFT loss) vs ${RL_RUN} (RL reward / pass@k)"
echo ""
echo "Expected behavior:"
echo "  - Prompt WITHOUT <|system|> tool block  -> writes code"
echo "  - Prompt WITH   <|system|> tool block   -> emits <functioncall> JSON"
echo ""
echo "Manual tests:"
echo "  ${PY} -m scripts.chat_cli    --ckpt ${RL_V6_CKPT}  --user 'write quicksort'"
echo "  ${PY} -m scripts.funcall_cli --ckpt ${RL_V6_CKPT}  --user 'Weather in Tokyo?'"
echo ""
echo "Evaluation:"
echo "  ${PY} -m scripts.eval_funcall --ckpt ${RL_V6_CKPT} --num-samples 16"
echo "  ${PY} -m scripts.eval_mbpp_pass_at_k --ckpt ${RL_V6_CKPT}    # if script exists"
echo ""
echo "Re-run partial stages:"
echo "  SKIP_TO=3 bash runs/train_a800_x8_v6.sh     # code+funcall data ready, merge"
echo "  SKIP_TO=4 bash runs/train_a800_x8_v6.sh     # merged data ready, SFT"
echo "  SKIP_TO=7 bash runs/train_a800_x8_v6.sh     # SFT ckpt + RL data ready, RL"
echo "  SKIP_TO=8 bash runs/train_a800_x8_v6.sh     # just the smoke test"
echo "  FORCE_PREP_CODE=1   ...                      # rebuild code SFT jsonl"
echo "  FORCE_PREP_FUNCALL=1 ...                     # rebuild funcall SFT jsonl"
echo "  FORCE_MERGE=1       ...                      # re-shuffle merged jsonl"
