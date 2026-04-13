#!/usr/bin/env bash
# =============================================================================
# CodeChat 8B pretraining pipeline — 8x A800-SXM4-80GB, bf16 + FSDP
#
# 相比 train_a800.sh / train_a800_v2.sh:
#   - 参数量 2B -> 8B (preset=8b, depth=40, n_embd=4096)
#   - 单卡 -> 8 卡 FSDP FULL_SHARD (参数/梯度/AdamW 状态全部切片)
#   - torchrun 启动，scripts.base_train 自动检测 LOCAL_RANK 并切到 FSDP 路径
#
# 为什么必须 FSDP (而不是 DDP):
#   - 8B fp32 AdamW 状态 ≈ 96GB，单张 80GB 卡根本放不下
#   - FSDP FULL_SHARD 把 params/grads/optim 按 rank 切成 8 份，单卡 ~12GB
#
# 关于 checkpoint:
#   - 2B 的 checkpoints/codechat_2b/latest.pt 维度和 8B 不兼容，无法直接续训
#   - 8B 走独立 run 名 codechat_8b，从零开始预训练
#
# 关于 Python 环境 (.venv_train):
#   - 宿主 python 3.12.3 + torch 2.10.0+cu130 是系统装好的，root 无法 pip 安装
#   - 脚本首次运行会在 .venv_train/ 建一个 venv，--system-site-packages 继承
#     系统的 torch / CUDA，再 pip 安装缺的 tensorboard / tiktoken / datasets / ...
#   - torchrun 走 `python -m torch.distributed.run`，避免 PATH 里找不到 torchrun
#     的入口脚本
#
# 前置条件:
#   - 8x A800 80GB (NVLink/NVSwitch 最佳)
#   - 系统预装 torch (>=2.1) 且能 import
#   - 预训练数据已准备好: data/pretrain/shard_*.bin
# =============================================================================
set -euo pipefail

export CODECHAT_DTYPE=bfloat16
export PYTHONUNBUFFERED=1
# NCCL tuning for A800 NVLink; safe defaults
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Avoid CUDA memory fragmentation with large FSDP blocks
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN=${RUN:-codechat_8b}
PRESET=${PRESET:-8b}
NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

# 跳过已完成阶段: SKIP_TO=4 会直接从 stage 4 开始
SKIP_TO=${SKIP_TO:-1}

# ---------------------------------------------------------------------------
# Bootstrap a training venv that inherits the system-installed torch.
#
# 这套宿主 (Debian/Ubuntu 风格) 有两个坑:
#   1) 系统装的 torch 往往在 /usr/local/lib/python3.12/dist-packages，
#      而 venv --system-site-packages 未必把 dist-packages 带进 sys.path
#   2) `python` 和 `python3` 可能是不同二进制，只有一个装了 torch
# 解决:
#   - 先探测哪个 python 能 import torch，用它建 venv
#   - 建完后再 sanity check，如果 venv 还是 import 不到 torch，就把宿主 torch
#     的 site-packages 目录写进 venv 的 .pth 文件里强行接上
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv_train"

# 1) 找一个能 import torch 的 system python
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

# 2) 记下宿主 torch 所在 site-packages，供后面 .pth 兜底
SYS_TORCH_SITE="$("${SYS_PYTHON}" -c 'import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))')"
echo "         torch site = ${SYS_TORCH_SITE}"

# 3) 建 venv
if [ ! -d "${VENV_DIR}" ]; then
    echo "==> [env] creating training venv at ${VENV_DIR}"
    echo "         (--system-site-packages so we reuse system torch/CUDA)"
    "${SYS_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --upgrade pip wheel
fi

PY="${VENV_DIR}/bin/python"

# 4) 兜底：如果 venv 还是找不到 torch，把宿主 torch site-packages 写进 .pth
if ! "${PY}" -c "import torch" >/dev/null 2>&1; then
    echo "==> [env] venv 无法 import torch，追加 .pth 指向宿主 torch site-packages"
    VENV_SITE="$("${PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    echo "${SYS_TORCH_SITE}" > "${VENV_SITE}/system_torch.pth"
fi

# 5) 安装训练需要但系统里可能没有的轻量包（幂等，已存在 venv 也会补装缺的）
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

# 6) 最终 sanity check
"${PY}" - <<'PYEOF'
import sys, torch
print(f"[env] python = {sys.executable}")
print(f"[env] torch  = {torch.__version__}  cuda={torch.version.cuda}  n_gpu={torch.cuda.device_count()}")
import tensorboard, tiktoken, numpy
print(f"[env] tensorboard={tensorboard.__version__}  tiktoken={tiktoken.__version__}  numpy={numpy.__version__}")
PYEOF

# 用 `python -m torch.distributed.run` 代替 `torchrun`，省得依赖 $PATH 里的 entry script
TORCHRUN_CMD=("${PY}" -m torch.distributed.run)

# 让 import codechat.* 能找到仓库根目录（venv 里没有以 editable 方式装本项目）
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ===========================================================================
# Stage 1: Pretrain data preparation
# ===========================================================================
if [ "$SKIP_TO" -le 1 ]; then
echo "==> [1/5] preparing pretraining shards"
# 首次运行取消注释 (16 个 shard ~ 8.1G, 如数据已就位可跳过):
##DONE# OUT_DIR=data/pretrain MAX_SHARDS=16 bash runs/prepare_pretrain_venv.sh
fi

# ===========================================================================
# Stage 2: Pretraining — 8B @ 8x A800 via FSDP
#
# 预算:
#   - device_batch_size=1, grad_accum=8, world_size=8
#   - global batch = 1 * 8 * 8 * 2048 = 131k tokens / step
#   - 30k steps ≈ 3.9B tokens seen (与 2B run 相当量级)
# ===========================================================================
### DONE: 预训练结束
##if [ "$SKIP_TO" -le 2 ]; then
##echo "==> [2/5] pretraining 8B (preset=$PRESET, FSDP x$NPROC)"
##"${TORCHRUN_CMD[@]}" \
##    --standalone \
##    --nproc_per_node="$NPROC" \
##    --master_addr="$MASTER_ADDR" \
##    --master_port="$MASTER_PORT" \
##    -m scripts.base_train \
##        --data-dir data/pretrain \
##        --preset "$PRESET" \
##        --block-size 2048 \
##        --device-batch-size 1 \
##        --grad-accum 8 \
##        --lr 1.5e-4 \
##        --warmup 1000 \
##        --max-steps 30000 \
##        --save-every 1000 \
##        --run-name "$RUN"
##fi

# ===========================================================================
# Stage 3: SFT data preparation
# ===========================================================================
if [ "$SKIP_TO" -le 3 ]; then
echo "==> [3/5] preparing SFT data"
### "${PY}" -m scripts.prepare_sft --out-dir data/sft ## DONE: 72M     data/sft/train.jsonl
fi

# ===========================================================================
# Stage 4: SFT — 8B also needs FSDP (optimizer states blow single card otherwise)
#
# 注意: scripts.chat_sft 当前还没加 FSDP 支持。若未打 FSDP 补丁，这一步
# 在 8B 会 OOM。可先用 d24 预设过 pipeline，或等 chat_sft.py 跟进分布式改动。 => DONE: https://github.com/xlisp/CodeChat/commit/bc8c26b1fa2c7322c0034bc18bf39167619a15b5
# ===========================================================================
# DONE: 16G     checkpoints/codechat_8b_sft/latest.pt
##if [ "$SKIP_TO" -le 4 ]; then
##echo "==> [4/5] SFT 8B (FSDP x$NPROC)"
##"${TORCHRUN_CMD[@]}" \
##    --standalone \
##    --nproc_per_node="$NPROC" \
##    --master_addr="$MASTER_ADDR" \
##    --master_port="$MASTER_PORT" \
##    -m scripts.chat_sft \
##        --base-ckpt "checkpoints/${RUN}/latest.pt" \
##        --data-dir data/sft \
##        --device-batch-size 1 \
##        --grad-accum 8 \
##        --max-steps 3000 \
##        --run-name "${RUN}_sft"
##fi

# ===========================================================================
# Stage 5: RL (GRPO on MBPP, executable reward)
# ===========================================================================
if [ "$SKIP_TO" -le 5 ]; then
echo "==> [5/5] RL — GRPO on MBPP (FSDP x$NPROC)"
"${TORCHRUN_CMD[@]}" \
    --standalone \
    --nproc_per_node="$NPROC" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m scripts.chat_rl \
        --sft-ckpt "checkpoints/${RUN}_sft/latest.pt" \
        --max-steps 1000 \
        --group-size 4 \
        --run-name "${RUN}_rl"
fi

# ===========================================================================
# Summary
# ===========================================================================
echo ""
echo "================================================================"
echo "  8B training (8x A800) complete!"
echo "================================================================"
echo ""
echo "Checkpoints:"
echo "  Pretrain: checkpoints/${RUN}/latest.pt"
echo "  SFT:      checkpoints/${RUN}_sft/latest.pt"
echo "  RL:       checkpoints/${RUN}_rl/latest.pt"
echo ""
echo "TensorBoard:"
echo "  ${VENV_DIR}/bin/tensorboard --logdir runs/tb"
echo ""
echo "Chat:"
echo "  ${PY} -m scripts.chat_cli --ckpt checkpoints/${RUN}_rl/latest.pt"
echo ""
echo "Skip to a specific stage on re-run:"
echo "  SKIP_TO=4 bash runs/train_a800_x8.sh"
