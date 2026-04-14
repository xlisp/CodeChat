#!/usr/bin/env bash
# =============================================================================
# CodeChat 8B — v3 code-boost + function-calling + (optional) MBPP RL
# 8x A800-SXM4-80GB, bf16 + FSDP
#
# 背景:
#   v2 (train_a800_x8_v2_funcall.sh) 发现 base ckpt checkpoints/codechat_8b_sft
#   在 MBPP 上 parseable 只有 ~1.75%, pass@8 = 0%，即便用 tiered reward + 过滤
#   题集也无法启动 GRPO —— "base model can't even produce parseable code"。
#
# v3 策略:
#   在 funcall SFT 之前先插一个 "代码 SFT 续训" 阶段，用 mbpp(非 train 切分) +
#   Codeforces-Python + the-stack-smol(python, 有 docstring 的子集) 继续
#   强化 codechat_8b_sft 的 instruction-to-Python 能力，产出新 ckpt
#   codechat_8b_sft_code。目标: parseable >= 50%, pass@8 >= 10%，满足后再跑
#   funcall SFT 和 RL 才有意义。
#
# 流水线:
#   [1] prepare code SFT data         -> data/sft_code/train.jsonl
#   [2] code SFT (FSDP x8)            -> checkpoints/codechat_8b_sft_code/latest.pt
#   [3] pass@k diagnostic on sft_code -> data/mbpp_passrate.jsonl
#       (打印 GATE: parseable>=50% & pass@8>=10% 才算达标)
#   [4] prepare funcall SFT data      -> data/sft_funcall/train.jsonl
#   [5] funcall SFT from sft_code     -> checkpoints/codechat_8b_funcall_v3/latest.pt
#   [RUN_RL=1 继续:]
#   [6] filter MBPP by pass rate      -> data/mbpp_rl_curriculum.jsonl
#   [7] optimized GRPO from sft_code  -> checkpoints/codechat_8b_rl_v3/latest.pt
#
# 为什么 RL base 仍用 sft_code 而不是 funcall ckpt:
#   funcall SFT 专门学 <functioncall> JSON 格式，会稀释代码能力。RL 的奖励
#   是可执行代码 pass 测试，和 funcall 格式无关，所以 base 用代码能力最强的
#   那个 ckpt 更合适。
#
# 跳过阶段:
#   SKIP_TO=2   跳过数据准备，直接从 code SFT 开始
#   SKIP_TO=3   已有 sft_code ckpt，从诊断开始
#   SKIP_TO=5   只跑 funcall SFT
#   RUN_RL=1    启用 RL (默认关闭)
#
# 前置条件:
#   - 8x A800 80GB
#   - checkpoints/codechat_8b_sft/latest.pt 已就位 (train_a800_x8.sh 产出)
#   - HF 可访问或已缓存: mbpp / MatrixStudio/Codeforces-Python-Submissions /
#     bigcode/the-stack-smol / glaiveai/glaive-function-calling-v2
# =============================================================================
set -euo pipefail

export CODECHAT_DTYPE=bfloat16
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN=${RUN:-codechat_8b}
CODE_RUN=${CODE_RUN:-${RUN}_sft_code}                       # 阶段 2 输出
FUNCALL_RUN=${FUNCALL_RUN:-${RUN}_funcall_v3}                # 阶段 5 输出
RL_RUN=${RL_RUN:-${RUN}_rl_v3}                               # 阶段 7 输出
BASE_CKPT=${BASE_CKPT:-checkpoints/${RUN}_sft/latest.pt}     # code SFT 起点

CODE_DATA_DIR=${CODE_DATA_DIR:-data/sft_code}
FUNCALL_DATA_DIR=${FUNCALL_DATA_DIR:-data/sft_funcall}
CODE_MAX_CF=${CODE_MAX_CF:-20000}
CODE_MAX_STACK=${CODE_MAX_STACK:-20000}
CODE_SOURCES=${CODE_SOURCES:-mbpp,codeforces,the-stack-smol}
FUNCALL_MAX_EXAMPLES=${FUNCALL_MAX_EXAMPLES:-0}

# code SFT 超参: 数据较小 (~40k) + 希望不破坏已学到的通用能力
CODE_LR=${CODE_LR:-2e-5}
CODE_MAX_STEPS=${CODE_MAX_STEPS:-2000}
CODE_WARMUP=${CODE_WARMUP:-100}

# funcall SFT 超参: 与 v2 一致 (113k 数据跑 4000 步)
FUNCALL_LR=${FUNCALL_LR:-3e-5}
FUNCALL_MAX_STEPS=${FUNCALL_MAX_STEPS:-4000}
FUNCALL_WARMUP=${FUNCALL_WARMUP:-200}

NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29502}   # 与 v1/v2 错开端口

SKIP_TO=${SKIP_TO:-1}

# ---- 诊断 + RL (默认关闭 RL，诊断总是跑) ----
RUN_RL=${RUN_RL:-0}
RL_N_DIAG=${RL_N_DIAG:-50}
RL_K_DIAG=${RL_K_DIAG:-8}
RL_MIN_PASS=${RL_MIN_PASS:-0.05}
RL_MAX_PASS=${RL_MAX_PASS:-0.95}
RL_GROUP_SIZE=${RL_GROUP_SIZE:-8}
RL_MAX_STEPS=${RL_MAX_STEPS:-500}
RL_LR=${RL_LR:-5e-6}

# 诊断门槛 —— 达不到就打印 WARN 但不强制中断 (方便手动继续 debug)
GATE_PARSEABLE=${GATE_PARSEABLE:-0.50}
GATE_PASS_AT_K=${GATE_PASS_AT_K:-0.10}

# ---------------------------------------------------------------------------
# Bootstrap training venv (与 v2 完全一致, 复用 .venv_train)
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
# 预检: base checkpoint
# ===========================================================================
if [ ! -f "${BASE_CKPT}" ]; then
    echo "ERROR: base checkpoint 不存在: ${BASE_CKPT}" >&2
    echo "       先跑完 runs/train_a800_x8.sh 的 SFT 阶段，或显式设置 BASE_CKPT。" >&2
    exit 1
fi
echo "==> [check] base ckpt = ${BASE_CKPT} ($(du -h "${BASE_CKPT}" | cut -f1))"

# ===========================================================================
# Stage 1: 组装 code SFT jsonl
# ===========================================================================
if [ "$SKIP_TO" -le 1 ]; then
    echo "==> [1/7] preparing extended code SFT data (mbpp non-train + codeforces + the-stack-smol)"
    if [ -s "${CODE_DATA_DIR}/train.jsonl" ] && [ "${FORCE_PREP_CODE:-0}" != "1" ]; then
        echo "    ${CODE_DATA_DIR}/train.jsonl 已存在 ($(wc -l < "${CODE_DATA_DIR}/train.jsonl") lines)，跳过。"
        echo "    重新生成: FORCE_PREP_CODE=1 bash runs/train_a800_x8_v3_funcall.sh"
    else
        "${PY}" -m scripts.prepare_sft_code \
            --out-dir "${CODE_DATA_DIR}" \
            --sources "${CODE_SOURCES}" \
            --max-codeforces "${CODE_MAX_CF}" \
            --max-the-stack "${CODE_MAX_STACK}"
    fi
fi

# ===========================================================================
# Stage 2: code SFT 续训 (FSDP x 8)
#
# 预算:
#   device_batch_size=1, grad_accum=8, world_size=8
#   global batch = 1 * 8 * 8 * 2048 ≈ 131k tokens / step
#   2000 steps ≈ 262M supervised tokens
#   lr 2e-5 (比 funcall 的 3e-5 更保守，避免破坏通用能力)
# ===========================================================================
if [ "$SKIP_TO" -le 2 ]; then
    echo "==> [2/7] code SFT (FSDP x${NPROC}, base=${BASE_CKPT} -> ${CODE_RUN})"
    "${TORCHRUN_CMD[@]}" \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m scripts.chat_sft \
            --base-ckpt "${BASE_CKPT}" \
            --data-dir "${CODE_DATA_DIR}" \
            --device-batch-size 1 \
            --grad-accum 8 \
            --lr "${CODE_LR}" \
            --warmup "${CODE_WARMUP}" \
            --max-steps "${CODE_MAX_STEPS}" \
            --run-name "${CODE_RUN}"
fi

CODE_CKPT="checkpoints/${CODE_RUN}/latest.pt"

# ===========================================================================
# Stage 3: pass@k 诊断 (在 sft_code ckpt 上，不在旧 sft ckpt 上)
#
# 这一步是 v3 的关键门槛：如果 parseable 还没到 50% / pass@8 没到 10%，说明
# 代码续训没到位，RL 仍然跑不起来。脚本打印 WARN 但继续 funcall stage，让
# 你可以手动决定是扩数据、加步数，还是放弃 RL。
# ===========================================================================
if [ "$SKIP_TO" -le 3 ]; then
    if [ ! -f "${CODE_CKPT}" ]; then
        echo "ERROR: code SFT ckpt 不存在: ${CODE_CKPT}" >&2
        exit 1
    fi
    echo "==> [3/7] MBPP pass@k diagnostic on code ckpt (ckpt=${CODE_CKPT}, n=${RL_N_DIAG}, k=${RL_K_DIAG})"
    mkdir -p data
    "${TORCHRUN_CMD[@]}" \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m scripts.eval_mbpp_pass_at_k \
            --ckpt "${CODE_CKPT}" \
            --split train \
            --n-problems "${RL_N_DIAG}" \
            --k "${RL_K_DIAG}" \
            --out-jsonl data/mbpp_passrate.jsonl

    # 不依赖输出格式：用 python 重算 gate (避免 grep 解析文本的脆弱性)
    echo "==> [3/7] checking GATE (parseable>=${GATE_PARSEABLE}, pass@${RL_K_DIAG}>=${GATE_PASS_AT_K})"
    "${PY}" - <<PYEOF
import json
rows = [json.loads(l) for l in open("data/mbpp_passrate.jsonl")]
if not rows:
    print("  GATE: no diagnostic rows — skipping gate check")
    raise SystemExit(0)
n = len(rows)
parseable = sum(1 for r in rows if (r.get("parseable_rate") or r.get("parseable") or 0) > 0) / n
# pass@k 在每题上由 "任一 sample 过" 表示
pass_at_k = sum(1 for r in rows if (r.get("pass_rate") or 0) > 0) / n
print(f"  problems={n}  parseable (any sample)={parseable:.3f}  pass@${RL_K_DIAG} (any)={pass_at_k:.3f}")
if parseable < ${GATE_PARSEABLE} or pass_at_k < ${GATE_PASS_AT_K}:
    print(f"  WARN: GATE NOT MET (need parseable>=${GATE_PARSEABLE} and pass@k>=${GATE_PASS_AT_K}).")
    print(f"        funcall SFT 可以照跑，但 RL 大概率仍会 reward=0。")
    print(f"        建议: 扩 CODE_MAX_CF/CODE_MAX_STACK，或把 CODE_MAX_STEPS 从 ${CODE_MAX_STEPS} 往上加。")
else:
    print("  GATE: OK — 可以进 funcall SFT 和 RL")
PYEOF
fi

# ===========================================================================
# Stage 4: funcall SFT jsonl (与 v2 相同，复用 data/sft_funcall)
# ===========================================================================
if [ "$SKIP_TO" -le 4 ]; then
    echo "==> [4/7] preparing function-calling SFT data"
    if [ -s "${FUNCALL_DATA_DIR}/train.jsonl" ] && [ "${FORCE_PREP_FUNCALL:-0}" != "1" ]; then
        echo "    ${FUNCALL_DATA_DIR}/train.jsonl 已存在 ($(wc -l < "${FUNCALL_DATA_DIR}/train.jsonl") lines)，跳过。"
        echo "    重新生成: FORCE_PREP_FUNCALL=1 bash runs/train_a800_x8_v3_funcall.sh"
    else
        "${PY}" -m scripts.prepare_sft_funcall \
            --out-dir "${FUNCALL_DATA_DIR}" \
            --max-examples "${FUNCALL_MAX_EXAMPLES}"
    fi
fi

# ===========================================================================
# Stage 5: funcall SFT, 起点是 code ckpt (不是原始 sft ckpt)
# ===========================================================================
if [ "$SKIP_TO" -le 5 ]; then
    if [ ! -f "${CODE_CKPT}" ]; then
        echo "ERROR: code SFT ckpt 不存在: ${CODE_CKPT}" >&2
        exit 1
    fi
    echo "==> [5/7] funcall SFT (FSDP x${NPROC}, base=${CODE_CKPT} -> ${FUNCALL_RUN})"
    "${TORCHRUN_CMD[@]}" \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m scripts.chat_sft \
            --base-ckpt "${CODE_CKPT}" \
            --data-dir "${FUNCALL_DATA_DIR}" \
            --device-batch-size 1 \
            --grad-accum 8 \
            --lr "${FUNCALL_LR}" \
            --warmup "${FUNCALL_WARMUP}" \
            --max-steps "${FUNCALL_MAX_STEPS}" \
            --run-name "${FUNCALL_RUN}"
fi

# ===========================================================================
# RL 阶段 (默认关闭)
# ===========================================================================
if [ "${RUN_RL}" != "1" ]; then
    echo ""
    echo "==> [skip] RL stages 6-7 disabled (pass RUN_RL=1 to enable)."
else

# ---------------------------------------------------------------------------
# Stage 6: 用 Stage 3 生成的 mbpp_passrate.jsonl 过滤题集
# ---------------------------------------------------------------------------
if [ "$SKIP_TO" -le 6 ]; then
    if [ ! -s data/mbpp_passrate.jsonl ]; then
        echo "ERROR: data/mbpp_passrate.jsonl 不存在，先跑 Stage 3。" >&2
        exit 1
    fi
    echo "==> [6/7] filter MBPP by pass rate in [${RL_MIN_PASS}, ${RL_MAX_PASS}]"
    "${PY}" -m scripts.filter_mbpp_by_passrate \
        --in-jsonl data/mbpp_passrate.jsonl \
        --out-jsonl data/mbpp_rl_curriculum.jsonl \
        --min-pass-rate "${RL_MIN_PASS}" \
        --max-pass-rate "${RL_MAX_PASS}"
fi

# ---------------------------------------------------------------------------
# Stage 7: 优化版 GRPO on MBPP，base = code ckpt (代码能力最强那个)
# ---------------------------------------------------------------------------
if [ "$SKIP_TO" -le 7 ]; then
    if [ ! -s data/mbpp_rl_curriculum.jsonl ]; then
        echo "ERROR: data/mbpp_rl_curriculum.jsonl 为空。说明 Stage 3 诊断里 group 内仍无方差。" >&2
        echo "       扩 code SFT 数据 / 步数，或先跑 RUN_RL=0 再分析 data/mbpp_passrate.jsonl。" >&2
        exit 1
    fi
    echo "==> [7/7] optimized MBPP GRPO (FSDP x${NPROC}, base=${CODE_CKPT} -> ${RL_RUN})"
    echo "    problems=$(wc -l < data/mbpp_rl_curriculum.jsonl)  "\
         "group=${RL_GROUP_SIZE}  reward=tiered  steps=${RL_MAX_STEPS}  lr=${RL_LR}"
    "${TORCHRUN_CMD[@]}" \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m scripts.chat_rl \
            --sft-ckpt "${CODE_CKPT}" \
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
echo "  v3 pipeline complete (RUN_RL=${RUN_RL})"
echo "================================================================"
echo ""
echo "Checkpoints:"
echo "  code SFT:    checkpoints/${CODE_RUN}/latest.pt"
echo "  funcall SFT: checkpoints/${FUNCALL_RUN}/latest.pt"
if [ "${RUN_RL}" = "1" ]; then
    echo "  MBPP RL v3:  checkpoints/${RL_RUN}/latest.pt"
fi
echo ""
echo "TensorBoard:"
echo "  ${VENV_DIR}/bin/tensorboard --logdir runs/tb"
echo "  (对比 codechat_8b_sft vs codechat_8b_sft_code 看 loss 曲线)"
echo ""
echo "Diagnostics:"
echo "  cat data/mbpp_passrate.jsonl   # 每题的 pass rate / tiered reward"
echo ""
echo "Quick smoke tests:"
echo "  ${PY} -m scripts.chat_cli --ckpt checkpoints/${CODE_RUN}/latest.pt"
echo "  ${PY} -m scripts.chat_cli --ckpt checkpoints/${FUNCALL_RUN}/latest.pt"
echo ""
echo "Re-run partial stages:"
echo "  SKIP_TO=2 bash runs/train_a800_x8_v3_funcall.sh        # 跳过数据, 从 code SFT 开始"
echo "  SKIP_TO=3 bash runs/train_a800_x8_v3_funcall.sh        # 已有 code ckpt, 跑诊断+funcall"
echo "  SKIP_TO=5 bash runs/train_a800_x8_v3_funcall.sh        # 只跑 funcall SFT"
echo "  RUN_RL=1 SKIP_TO=6 bash runs/train_a800_x8_v3_funcall.sh   # 仅跑过滤+RL"
echo "  FORCE_PREP_CODE=1 bash runs/train_a800_x8_v3_funcall.sh    # 重建 code jsonl"
