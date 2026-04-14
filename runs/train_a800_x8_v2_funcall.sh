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
#   - 跳过 pretrain / 通用 SFT，默认只做 "funcall 续 SFT"
#   - 基线 ckpt 默认指向 codechat_8b_sft (可通过 BASE_CKPT 覆盖)
#   - 新 run 名 codechat_8b_funcall，权重不会覆盖 sft checkpoint
#
# 可选: RUN_RL=1 启用"救活 MBPP RL"的 3 个补充阶段 (详见 5.6 优化清单 #1/#3/#5/#6)
#   Stage 3 — pass@k 诊断: 量化 SFT ckpt 在 MBPP 上的真实能力
#   Stage 4 — 按 pass rate 过滤 MBPP: 只保留 group 内能有方差的题
#   Stage 5 — 优化版 GRPO: tiered reward + group_size=8 + 每 50 步打印 rollout
# 如果 Stage 3 的 pass@1 < 1%，脚本会打印 WARN 但仍继续 (你可以手动中断)，
# 因为此时即便 tiered reward 也很难看到 reward 曲线离开 0。
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

# 跳过已完成阶段: SKIP_TO=2 直接开始 SFT，SKIP_TO=3 开始 RL 诊断
SKIP_TO=${SKIP_TO:-1}

# ---- 可选 RL 阶段 (默认关闭；开启后跑 §5.6 优化版 MBPP RL) ----
RUN_RL=${RUN_RL:-0}
RL_BASE_CKPT=${RL_BASE_CKPT:-checkpoints/${RUN}_sft/latest.pt}  # RL 建议从通用 SFT 起，别用 funcall ckpt
RL_RUN=${RL_RUN:-${RUN}_rl_v2}
RL_N_DIAG=${RL_N_DIAG:-50}          # pass@k 诊断评测题数
RL_K_DIAG=${RL_K_DIAG:-8}           # 每题 K 个样本
RL_MIN_PASS=${RL_MIN_PASS:-0.05}    # 过滤 MBPP: 保留 pass_rate ∈ [MIN, MAX]
RL_MAX_PASS=${RL_MAX_PASS:-0.95}
RL_GROUP_SIZE=${RL_GROUP_SIZE:-8}
RL_MAX_STEPS=${RL_MAX_STEPS:-500}
RL_LR=${RL_LR:-5e-6}                # 比旧 RL 的 1e-5 更保守，tiered reward 下梯度幅值更大

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
    echo "==> [2/5] function-calling SFT (FSDP x${NPROC}, base=${BASE_CKPT})"
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
# 以下 Stage 3-5 为"救活 MBPP RL"的优化版流程，默认关闭。
# 开启: RUN_RL=1 bash runs/train_a800_x8_v2_funcall.sh
#
# 与 train_a800_x8.sh 里失败的那一版 RL 的区别:
#   - reward 从 binary pass/fail 换成 tiered 阶梯:
#       syntax 错 0.00 / 运行报错 0.05 / 能跑 0 test 过 0.15 / k/n 过
#       0.15+0.85*(k/n) / 全过 1.00   → group 内奖励几乎必有方差
#   - group_size 4 → 8: 弱基线下仍能偶遇 1 个 rollout 更接近
#   - 先跑 pass@k 诊断 + 按 pass rate 过滤 MBPP，把 "group 全 0" 的题去掉
#   - 每 50 步打印一个 rollout 文本到 stdout 和 TB，肉眼可见模型在写什么
# ===========================================================================

if [ "${RUN_RL}" != "1" ]; then
    echo ""
    echo "==> [skip] RL stages 3-5 disabled (pass RUN_RL=1 to enable)."
else

# ---------------------------------------------------------------------------
# Stage 3: pass@k 诊断 (优化清单 #6)
#
# 在动手训练前先量化 SFT ckpt 在 MBPP 上的实际能力，避免再次陷入 reward=0
# 死循环。产出 data/mbpp_passrate.jsonl，记录每题的 pass rate / tiered reward。
# ---------------------------------------------------------------------------
if [ "$SKIP_TO" -le 3 ]; then
    if [ ! -f "${RL_BASE_CKPT}" ]; then
        echo "ERROR: RL base ckpt 不存在: ${RL_BASE_CKPT}" >&2
        exit 1
    fi
    echo "==> [3/5] MBPP pass@k diagnostic (ckpt=${RL_BASE_CKPT}, n=${RL_N_DIAG}, k=${RL_K_DIAG})"
    mkdir -p data
    "${TORCHRUN_CMD[@]}" \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m scripts.eval_mbpp_pass_at_k \
            --ckpt "${RL_BASE_CKPT}" \
            --split train \
            --n-problems "${RL_N_DIAG}" \
            --k "${RL_K_DIAG}" \
            --out-jsonl data/mbpp_passrate.jsonl
fi

# ---------------------------------------------------------------------------
# Stage 4: 按 pass rate 过滤 MBPP (优化清单 #3)
#
# 只保留 pass rate ∈ [RL_MIN_PASS, RL_MAX_PASS] 的题目。GRPO 只能从 group 内
# 方差里学，全 0 或全 1 的题都浪费步数。
# ---------------------------------------------------------------------------
if [ "$SKIP_TO" -le 4 ]; then
    echo "==> [4/5] filter MBPP by pass rate in [${RL_MIN_PASS}, ${RL_MAX_PASS}]"
    "${PY}" -m scripts.filter_mbpp_by_passrate \
        --in-jsonl data/mbpp_passrate.jsonl \
        --out-jsonl data/mbpp_rl_curriculum.jsonl \
        --min-pass-rate "${RL_MIN_PASS}" \
        --max-pass-rate "${RL_MAX_PASS}"
fi

# ---------------------------------------------------------------------------
# Stage 5: 优化版 GRPO on MBPP (优化清单 #1 + #4 + #5)
#
# 关键参数:
#   --reward-mode tiered      # §5.6 #1: 阶梯奖励打破全 0
#   --group-size 8            # §5.6 #4: 2 倍于旧 run，拉高至少一个非零 rollout 的概率
#   --problems-file ...       # §5.6 #3: 只在过滤后的题集上训练
#   --log-rollouts-every 50   # §5.6 #5: 每 50 步 dump 一个 rollout，看模型到底在写啥
#
# 注意: RL run name = ${RL_RUN} (默认 codechat_8b_rl_v2)，不会覆盖旧的
# codechat_8b_rl/latest.pt，两次跑的 TB 可直接对比。
# ---------------------------------------------------------------------------
if [ "$SKIP_TO" -le 5 ]; then
    if [ ! -s data/mbpp_rl_curriculum.jsonl ]; then
        echo "ERROR: data/mbpp_rl_curriculum.jsonl 为空，不能开 RL。" >&2
        echo "       说明 SFT base 能力过低 (见 Stage 3 VERDICT)，先强化 base。" >&2
        exit 1
    fi
    echo "==> [5/5] optimized MBPP GRPO (FSDP x${NPROC}, base=${RL_BASE_CKPT})"
    echo "    problems=$(wc -l < data/mbpp_rl_curriculum.jsonl)  "\
         "group=${RL_GROUP_SIZE}  reward=tiered  steps=${RL_MAX_STEPS}  lr=${RL_LR}"
    "${TORCHRUN_CMD[@]}" \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m scripts.chat_rl \
            --sft-ckpt "${RL_BASE_CKPT}" \
            --problems-file data/mbpp_rl_curriculum.jsonl \
            --reward-mode tiered \
            --group-size "${RL_GROUP_SIZE}" \
            --max-steps "${RL_MAX_STEPS}" \
            --lr "${RL_LR}" \
            --log-rollouts-every 50 \
            --run-name "${RL_RUN}"
fi

fi  # RUN_RL

# ===========================================================================
# Summary
# ===========================================================================
echo ""
echo "================================================================"
echo "  pipeline complete (RUN_RL=${RUN_RL})"
echo "================================================================"
echo ""
echo "Checkpoints:"
echo "  funcall SFT: checkpoints/${FUNCALL_RUN}/latest.pt"
if [ "${RUN_RL}" = "1" ]; then
    echo "  MBPP RL v2:  checkpoints/${RL_RUN}/latest.pt"
fi
echo ""
echo "TensorBoard:"
echo "  ${VENV_DIR}/bin/tensorboard --logdir runs/tb"
echo "  (比较 codechat_8b_rl vs ${RL_RUN} 看 reward 曲线是否摆脱了恒 0)"
echo ""
echo "Quick smoke test:"
echo "  ${PY} -m scripts.chat_cli --ckpt checkpoints/${FUNCALL_RUN}/latest.pt"
echo ""
echo "Re-run partial stages:"
echo "  SKIP_TO=2 bash runs/train_a800_x8_v2_funcall.sh               # 只跑 funcall SFT"
echo "  FORCE_PREP=1 bash runs/train_a800_x8_v2_funcall.sh            # 重建 jsonl"
echo "  MAX_EXAMPLES=2000 bash runs/train_a800_x8_v2_funcall.sh       # 小规模 smoke"
echo "  RUN_RL=1 SKIP_TO=3 bash runs/train_a800_x8_v2_funcall.sh      # 仅跑 RL 3 阶段"
echo "  RUN_RL=1 SKIP_TO=5 bash runs/train_a800_x8_v2_funcall.sh      # 跳过诊断/过滤 (已有 curriculum)"
