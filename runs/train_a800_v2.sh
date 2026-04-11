#!/usr/bin/env bash
# =============================================================================
# CodeChat v2 training pipeline — single A800-SXM4-80GB, bf16
#
# v2 在 v1 (train_a800.sh) 基础上追加:
#   - SWE-bench Lite 基线评测
#   - GRPO on SWE-bench train (三阶段课程学习: syntax → apply → docker)
#   - 训练后再次评测，量化 SWE-bench RL 的增益
#
# 前置条件 (SWE-bench RL 阶段):
#   - Docker: docker info 能无 sudo 运行
#   - pip install swebench
#   - ~200-400GB 磁盘用于 Docker 镜像
#   - Phase 3 (docker) 非常慢，每步 1-5 分钟
#
# 详细说明见 docs/swebench_rl.md
# =============================================================================
set -euo pipefail

export CODECHAT_DTYPE=bfloat16
export PYTHONUNBUFFERED=1

RUN=${RUN:-codechat_2b}
PRESET=${PRESET:-2b}

# 跳过已完成阶段: SKIP_TO=6 会直接从 stage 6 开始
SKIP_TO=${SKIP_TO:-1}

# ===========================================================================
# Stage 1: Pretrain data preparation
# ===========================================================================
if [ "$SKIP_TO" -le 1 ]; then
echo "==> [1/8] preparing pretraining shards"
# 首次运行取消注释:
### OUT_DIR=data/pretrain MAX_SHARDS=16 bash runs/prepare_pretrain_venv.sh
fi

# ===========================================================================
# Stage 2: Pretraining
# ===========================================================================
if [ "$SKIP_TO" -le 2 ]; then
echo "==> [2/8] pretraining (preset=$PRESET, ~2B params)"
python -m scripts.base_train \
    --data-dir data/pretrain \
    --preset "$PRESET" \
    --device-batch-size 2 \
    --grad-accum 16 \
    --max-steps 30000 \
    --run "$RUN"
fi

# ===========================================================================
# Stage 3: SFT data preparation
# ===========================================================================
if [ "$SKIP_TO" -le 3 ]; then
echo "==> [3/8] preparing SFT data"
python -m scripts.prepare_sft --out-dir data/sft
fi

# ===========================================================================
# Stage 4: Supervised Fine-Tuning
# ===========================================================================
if [ "$SKIP_TO" -le 4 ]; then
echo "==> [4/8] SFT"
python -m scripts.chat_sft \
    --base-ckpt "checkpoints/${RUN}/latest.pt" \
    --data-dir data/sft \
    --device-batch-size 1 \
    --grad-accum 16 \
    --max-steps 3000 \
    --run "${RUN}_sft"
fi

# ===========================================================================
# Stage 5: GRPO on MBPP (executable reward, 基础代码能力)
# ===========================================================================
if [ "$SKIP_TO" -le 5 ]; then
echo "==> [5/8] RL — GRPO on MBPP (executable reward)"
python -m scripts.chat_rl \
    --sft-ckpt "checkpoints/${RUN}_sft/latest.pt" \
    --max-steps 1000 \
    --group-size 4 \
    --run "${RUN}_rl"
fi

# ===========================================================================
# Stage 6: SWE-bench Lite 基线评测 (patch generation only, 不需要 Docker)
# ===========================================================================
if [ "$SKIP_TO" -le 6 ]; then
echo "==> [6/8] SWE-bench Lite baseline eval (before SWE-bench RL)"
python -m scripts.eval_swebench \
    --ckpt "checkpoints/${RUN}_rl/latest.pt" \
    --split lite \
    --out "predictions/${RUN}_rl_baseline.jsonl" \
    --model-name "${RUN}-rl-baseline" \
    --limit 50
fi

# ===========================================================================
# Stage 7: GRPO on SWE-bench train (三阶段课程学习)
#
#   Phase 1 — syntax:     教 diff 格式 (无 Docker, 快)
#   Phase 2 — apply-only: 教补丁 apply (Docker, 中速)
#   Phase 3 — docker:     教修复 bug   (Docker, 慢)
#
# 详细原理见 docs/swebench_rl.md
# ===========================================================================
if [ "$SKIP_TO" -le 7 ]; then
echo "==> [7/8] RL — GRPO on SWE-bench train (Docker-in-the-Loop)"

# --- Pre-build Docker images (only needed once, skip if already built) ---
echo "  [7a] checking / building Docker images ..."
python -m scripts.rl_swebench \
    --sft-ckpt "checkpoints/${RUN}_rl/latest.pt" \
    --prepare-images \
    --max-steps 0

# --- Phase 1: syntax warm-up ---
# 模型学习: 输出合法的 unified diff 格式 (有 @@ hunk header)
# 速度: ~5s/step, 总计 ~17 min
echo "  [7b] Phase 1: syntax warm-up (200 steps)"
python -m scripts.rl_swebench \
    --sft-ckpt "checkpoints/${RUN}_rl/latest.pt" \
    --reward-mode syntax \
    --max-steps 200 \
    --group-size 4 \
    --lr 5e-6 \
    --run "${RUN}_rl_sweb_syntax"

# --- Phase 2: apply-only ---
# 模型学习: 生成能在真实仓库上 git apply 的补丁 (正确的路径、行号、上下文)
# 速度: ~15s/step, 总计 ~75 min
echo "  [7c] Phase 2: apply-only (300 steps)"
python -m scripts.rl_swebench \
    --sft-ckpt "checkpoints/${RUN}_rl_sweb_syntax/latest.pt" \
    --reward-mode apply-only \
    --max-steps 300 \
    --group-size 4 \
    --docker-workers 4 \
    --lr 3e-6 \
    --run "${RUN}_rl_sweb_apply"

# --- Phase 3: full Docker evaluation ---
# 模型学习: 生成能修复 bug 的补丁 (FAIL_TO_PASS 测试通过)
# 速度: ~2-5 min/step, 总计 ~17-42 h (这是整个流水线最慢的部分)
echo "  [7d] Phase 3: full Docker eval (500 steps)"
python -m scripts.rl_swebench \
    --sft-ckpt "checkpoints/${RUN}_rl_sweb_apply/latest.pt" \
    --reward-mode docker \
    --max-steps 500 \
    --group-size 4 \
    --docker-workers 4 \
    --docker-timeout 300 \
    --lr 2e-6 \
    --run "${RUN}_rl_sweb"
fi

# ===========================================================================
# Stage 8: SWE-bench Lite 最终评测 (对比 stage 6 的基线)
# ===========================================================================
if [ "$SKIP_TO" -le 8 ]; then
echo "==> [8/8] SWE-bench Lite final eval (after SWE-bench RL)"
python -m scripts.eval_swebench \
    --ckpt "checkpoints/${RUN}_rl_sweb/latest.pt" \
    --split lite \
    --out "predictions/${RUN}_rl_sweb.jsonl" \
    --model-name "${RUN}-rl-sweb" \
    --limit 50
fi

# ===========================================================================
# Summary
# ===========================================================================
echo ""
echo "================================================================"
echo "  v2 training complete!"
echo "================================================================"
echo ""
echo "Checkpoints:"
echo "  MBPP RL:    checkpoints/${RUN}_rl/latest.pt"
echo "  SWE-bench:  checkpoints/${RUN}_rl_sweb/latest.pt"
echo ""
echo "Predictions:"
echo "  Baseline:   predictions/${RUN}_rl_baseline.jsonl"
echo "  After SWEB: predictions/${RUN}_rl_sweb.jsonl"
echo ""
echo "Compare baseline vs final:"
echo "  diff <(grep -c '\"model_patch\": \"\"' predictions/${RUN}_rl_baseline.jsonl) \\"
echo "       <(grep -c '\"model_patch\": \"\"' predictions/${RUN}_rl_sweb.jsonl)"
echo ""
echo "Run official harness (requires Docker on eval machine):"
echo "  python -m swebench.harness.run_evaluation \\"
echo "      --dataset_name princeton-nlp/SWE-bench_Lite \\"
echo "      --predictions_path predictions/${RUN}_rl_sweb.jsonl \\"
echo "      --max_workers 4 --run_id ${RUN}_rl_sweb"
echo ""
echo "TensorBoard:"
echo "  tensorboard --logdir runs/tb"
echo ""
echo "Chat:"
echo "  python -m scripts.chat_cli --ckpt checkpoints/${RUN}_rl_sweb/latest.pt"
echo ""
echo "Skip to a specific stage on re-run:"
echo "  SKIP_TO=7 bash runs/train_a800_v2.sh"
