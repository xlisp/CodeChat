#!/usr/bin/env bash
# =============================================================================
# CodeChat 8B — function-calling SFT pipeline (8x A800-SXM4-80GB, bf16 + FSDP)
#
# 目标:
#   在已经训练好的 checkpoints/codechat_8b_sft/latest.pt 基础上，继续 SFT，
#   让模型学会 function calling: 在看到 SYSTEM 里声明的函数 + USER 请求后，
#   输出 <functioncall> {"name": ..., "arguments": ...} 这样的 JSON 调用；
#   收到 FUNCTION RESPONSE 后再用自然语言总结给用户。
#
# 数据:
#   glaiveai/glaive-function-calling-v2 (~113k 条多轮对话)
#
# 格式约定 (复用 codechat 已有 chat tag，不改 tokenizer):
#   <|system|>\n{function schema + prompt}\n<|end|>\n
#   <|user|>\n{user turn}\n<|end|>\n
#   <|assistant|>\n<functioncall> {...json...}\n<|end|>\n
#   <|function_response|>\n{tool output}\n<|end|>\n
#   <|assistant|>\n{final natural-language answer}\n<|end|>\n
#
#   loss 只计算 assistant turn 的 token (system / user / function_response
#   全部 -100 掩掉)，让模型专门学: 该在什么时候发 <functioncall>、
#   以及 function response 回来之后怎么措辞。
#
# 相对 train_a800_x8.sh:
#   - 跳过 pretrain / 通用 SFT / RL，只做 "funcall 续 SFT"
#   - 基线 ckpt 默认指向 codechat_8b_sft (可通过 BASE_CKPT 覆盖)
#   - 新 run 名 codechat_8b_funcall，权重不会覆盖 sft checkpoint
#
# 前置条件:
#   - 8x A800 80GB
#   - checkpoints/codechat_8b_sft/latest.pt 已就位 (由 train_a800_x8.sh 产出)
#   - HF 能访问 (或预先 HF_HOME 里缓存好 glaive-function-calling-v2)
# =============================================================================
set -euo pipefail

export CODECHAT_DTYPE=bfloat16
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN=${RUN:-codechat_8b}
FUNCALL_RUN=${FUNCALL_RUN:-${RUN}_funcall}
BASE_CKPT=${BASE_CKPT:-checkpoints/${RUN}_sft/latest.pt}
DATA_DIR=${DATA_DIR:-data/sft_funcall}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}        # 0 = use all ~113k rows

NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}      # 与 x8.sh 错开，避免同机并跑撞端口

# 跳过已完成阶段: SKIP_TO=2 直接开始 SFT (前提: data/sft_funcall/train.jsonl 已生成)
SKIP_TO=${SKIP_TO:-1}

# ---------------------------------------------------------------------------
# Bootstrap training venv — 完全复用 train_a800_x8.sh 的逻辑
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
    echo "ERROR: 找不到装了 torch 的 system python。请手动 SYS_PYTHON=/path/to/python 重试。" >&2
    exit 1
fi
echo "==> [env] system python = ${SYS_PYTHON}"
"${SYS_PYTHON}" -c "import torch, sys; print(f'         -> python {sys.version.split()[0]}, torch {torch.__version__}, cuda {torch.version.cuda}')"

SYS_TORCH_SITE="$("${SYS_PYTHON}" -c 'import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))')"
echo "         torch site = ${SYS_TORCH_SITE}"

if [ ! -d "${VENV_DIR}" ]; then
    echo "==> [env] creating training venv at ${VENV_DIR}"
    "${SYS_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --upgrade pip wheel
fi

PY="${VENV_DIR}/bin/python"

if ! "${PY}" -c "import torch" >/dev/null 2>&1; then
    echo "==> [env] venv 无法 import torch，追加 .pth 指向宿主 torch site-packages"
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
import sys, torch
print(f"[env] python = {sys.executable}")
print(f"[env] torch  = {torch.__version__}  cuda={torch.version.cuda}  n_gpu={torch.cuda.device_count()}")
import tensorboard, tiktoken, numpy, datasets
print(f"[env] tensorboard={tensorboard.__version__}  tiktoken={tiktoken.__version__}  "
      f"numpy={numpy.__version__}  datasets={datasets.__version__}")
PYEOF

TORCHRUN_CMD=("${PY}" -m torch.distributed.run)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ===========================================================================
# 预检: 确认 base checkpoint 存在
# ===========================================================================
if [ ! -f "${BASE_CKPT}" ]; then
    echo "ERROR: base checkpoint 不存在: ${BASE_CKPT}" >&2
    echo "       先跑完 runs/train_a800_x8.sh 的 SFT 阶段，或显式设置 BASE_CKPT。" >&2
    exit 1
fi
echo "==> [check] base ckpt = ${BASE_CKPT} ($(du -h "${BASE_CKPT}" | cut -f1))"

# ===========================================================================
# Stage 1: 从 glaive-function-calling-v2 生成 SFT jsonl
#
# 产出: ${DATA_DIR}/train.jsonl —— 每行 {"input_ids": [...], "labels": [...]}
# MAX_EXAMPLES=0 表示全量 (~113k)，想快速 smoke test 可传 MAX_EXAMPLES=2000
# ===========================================================================
if [ "$SKIP_TO" -le 1 ]; then
    echo "==> [1/2] preparing function-calling SFT data (glaiveai/glaive-function-calling-v2)"
    if [ -s "${DATA_DIR}/train.jsonl" ] && [ "${FORCE_PREP:-0}" != "1" ]; then
        echo "    ${DATA_DIR}/train.jsonl 已存在 ($(wc -l < "${DATA_DIR}/train.jsonl") lines)，跳过。"
        echo "    重新生成: FORCE_PREP=1 bash runs/train_a800_x8_v2_funcall.sh"
    else
        "${PY}" -m scripts.prepare_sft_funcall \
            --out-dir "${DATA_DIR}" \
            --max-examples "${MAX_EXAMPLES}"
    fi
fi

# ===========================================================================
# Stage 2: Function-calling SFT (FSDP x 8)
#
# 预算:
#   - device_batch_size=1, grad_accum=8, world_size=8
#   - global batch = 1 * 8 * 8 * 2048 ≈ 131k tokens / step
#   - 4000 steps ≈ 524M supervised tokens，够过数据集 ~4-5 遍
#   - lr 3e-5 (略低于通用 SFT 的 5e-5，避免破坏已对齐的通用能力)
# ===========================================================================
if [ "$SKIP_TO" -le 2 ]; then
    echo "==> [2/2] function-calling SFT (FSDP x${NPROC}, base=${BASE_CKPT})"
    "${TORCHRUN_CMD[@]}" \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m scripts.chat_sft \
            --base-ckpt "${BASE_CKPT}" \
            --data-dir "${DATA_DIR}" \
            --device-batch-size 1 \
            --grad-accum 8 \
            --lr 3e-5 \
            --warmup 200 \
            --max-steps 4000 \
            --run-name "${FUNCALL_RUN}"
fi

# ===========================================================================
# Summary
# ===========================================================================
echo ""
echo "================================================================"
echo "  function-calling SFT complete!"
echo "================================================================"
echo ""
echo "Checkpoint:"
echo "  ${FUNCALL_RUN}: checkpoints/${FUNCALL_RUN}/latest.pt"
echo ""
echo "TensorBoard:"
echo "  ${VENV_DIR}/bin/tensorboard --logdir runs/tb"
echo ""
echo "Quick smoke test:"
echo "  ${PY} -m scripts.chat_cli --ckpt checkpoints/${FUNCALL_RUN}/latest.pt"
echo ""
echo "Re-run partial stages:"
echo "  SKIP_TO=2 bash runs/train_a800_x8_v2_funcall.sh      # 只跑 SFT"
echo "  FORCE_PREP=1 bash runs/train_a800_x8_v2_funcall.sh   # 重建 jsonl"
echo "  MAX_EXAMPLES=2000 bash runs/train_a800_x8_v2_funcall.sh  # 小规模 smoke"
