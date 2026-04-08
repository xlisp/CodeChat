#!/usr/bin/env bash
# End-to-end CodeChat training on a single A800-SXM4-80GB, bf16.
set -euo pipefail

export CODECHAT_DTYPE=bfloat16
export PYTHONUNBUFFERED=1

RUN=${RUN:-codechat_2b}
PRESET=${PRESET:-2b}

echo "==> [1/5] preparing pretraining shards"
OUT_DIR=data/pretrain MAX_SHARDS=16 bash runs/prepare_pretrain_venv.sh

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

echo "==> done. try it:"
echo "    python -m scripts.chat_cli --ckpt checkpoints/${RUN}_rl/latest.pt"
