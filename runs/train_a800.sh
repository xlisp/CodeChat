#!/usr/bin/env bash
# End-to-end CodeChat training on a single A800-SXM4-80GB, bf16.
set -euo pipefail

export CODECHAT_DTYPE=bfloat16
export PYTHONUNBUFFERED=1

RUN=${RUN:-codechat_2b}
PRESET=${PRESET:-2b}

echo "==> [1/5] preparing pretraining shards"
# single execute: DONE:  256M    data/pretrain/shard_0000.bin ... data/pretrain/shard_0031.bin => all is 8.1G
### OUT_DIR=data/pretrain MAX_SHARDS=16 bash runs/prepare_pretrain_venv.sh

echo "==> [2/5] pretraining (preset=$PRESET, ~2B params)"
python -m scripts.base_train \
    --data-dir data/pretrain \
    --preset "$PRESET" \
    --device-batch-size 2 \
    --grad-accum 16 \
    --max-steps 30000 \
    --run "$RUN"

echo "==> [3/5] preparing SFT data"
python -m scripts.prepare_sft --out-dir data/sft

echo "==> [4/5] SFT"
python -m scripts.chat_sft \
    --base-ckpt "checkpoints/${RUN}/latest.pt" \
    --data-dir data/sft \
    --device-batch-size 1 \
    --grad-accum 16 \
    --max-steps 3000 \
    --run "${RUN}_sft"

echo "==> [5/5] RL (GRPO on MBPP, executable reward)"
python -m scripts.chat_rl \
    --sft-ckpt "checkpoints/${RUN}_sft/latest.pt" \
    --max-steps 1000 \
    --group-size 4 \
    --run "${RUN}_rl"

#-# echo "==> [6/7] SWE-bench Lite eval (patch generation only)"
# python -m scripts.eval_swebench \
#    --ckpt "checkpoints/${RUN}_rl/latest.pt" \
#    --split lite \
#    --out "predictions/${RUN}_rl.jsonl" \
#    --model-name "${RUN}-rl" \
#    --limit 50

# --- Stage 7: GRPO on SWE-bench train (Docker required, very heavy) ---
# Uncomment to enable. Requires:
#   - Docker installed and accessible (docker info)
#   - pip install swebench
#   - Pre-built Docker images (run once with --prepare-images --max-steps 0)
#   - ~120GB disk for Docker images
#   - Expect ~1-5 min per RL step in "docker" mode
#
# Recommended: run the three phases sequentially (syntax -> apply -> docker)
#
# echo "==> [7/7] GRPO on SWE-bench train (Docker RL)"
# # Phase 1: syntax warm-up (teach diff format, fast, no Docker)
# python -m scripts.rl_swebench \
#     --sft-ckpt "checkpoints/${RUN}_rl/latest.pt" \
#     --reward-mode syntax \
#     --max-steps 200 \
#     --group-size 4 \
#     --run "${RUN}_rl_sweb_syntax"
#
# # Phase 2: apply-only (teach valid patches, medium speed)
# python -m scripts.rl_swebench \
#     --sft-ckpt "checkpoints/${RUN}_rl_sweb_syntax/latest.pt" \
#     --reward-mode apply-only \
#     --max-steps 300 \
#     --group-size 4 \
#     --docker-workers 4 \
#     --run "${RUN}_rl_sweb_apply"
#
# # Phase 3: full Docker eval (teach test-passing patches, slow)
# python -m scripts.rl_swebench \
#     --sft-ckpt "checkpoints/${RUN}_rl_sweb_apply/latest.pt" \
#     --reward-mode docker \
#     --max-steps 500 \
#     --group-size 4 \
#     --docker-workers 4 \
#     --docker-timeout 300 \
#     --run "${RUN}_rl_sweb"

echo "==> done. try it:"
echo "    python -m scripts.chat_cli --ckpt checkpoints/${RUN}_rl/latest.pt"
