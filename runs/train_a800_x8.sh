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
#   - 如果想从 2B 蒸馏或 warm-start 另说 (--resume 只接受同 shape)
#
# 前置条件:
#   - 8x A800 80GB (NVLink/NVSwitch 最佳)
#   - torch>=2.1 (FSDP use_orig_params=True 需要)
#   - CUDA 12.x, NCCL 可用
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

# ===========================================================================
# Stage 1: Pretrain data preparation
# ===========================================================================
if [ "$SKIP_TO" -le 1 ]; then
echo "==> [1/5] preparing pretraining shards"
# 首次运行取消注释 (16 个 shard ~ 8.1G, 如数据已就位可跳过):
### OUT_DIR=data/pretrain MAX_SHARDS=16 bash runs/prepare_pretrain_venv.sh
fi

# ===========================================================================
# Stage 2: Pretraining — 8B @ 8x A800 via FSDP
#
# 预算:
#   - device_batch_size=1, grad_accum=8, world_size=8
#   - global batch = 1 * 8 * 8 * 2048 = 131k tokens / step
#   - 30k steps ≈ 3.9B tokens seen (与 2B run 相当量级)
#
# 如果显存告急，先降 device-batch-size 到 1 并加大 grad-accum；
# 若还不够，可以调整 block_size=1024 (但数据也得改)
# ===========================================================================
if [ "$SKIP_TO" -le 2 ]; then
echo "==> [2/5] pretraining 8B (preset=$PRESET, FSDP x$NPROC)"
torchrun \
    --standalone \
    --nproc_per_node="$NPROC" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m scripts.base_train \
        --data-dir data/pretrain \
        --preset "$PRESET" \
        --block-size 2048 \
        --device-batch-size 1 \
        --grad-accum 8 \
        --lr 1.5e-4 \
        --warmup 1000 \
        --max-steps 30000 \
        --save-every 1000 \
        --run-name "$RUN"
fi

# ===========================================================================
# Stage 3: SFT data preparation
# ===========================================================================
if [ "$SKIP_TO" -le 3 ]; then
echo "==> [3/5] preparing SFT data"
python -m scripts.prepare_sft --out-dir data/sft
fi

# ===========================================================================
# Stage 4: SFT — 8B also needs FSDP (optimizer states blow single card otherwise)
#
# 注意: scripts.chat_sft 尚未加 FSDP 支持。这里留两种选择：
#   (a) 若已给 chat_sft.py 打上和 base_train 一样的 FSDP 补丁，直接 torchrun 即可
#   (b) 若暂未实现，可以用更小的 lr + 多卡 DDP 跑短 SFT (不推荐 8B 单卡)
# 当前脚本按 (a) 写，后续在 chat_sft.py 里补 setup_distributed+wrap_fsdp 即可复用
# ===========================================================================
if [ "$SKIP_TO" -le 4 ]; then
echo "==> [4/5] SFT 8B (FSDP x$NPROC)"
torchrun \
    --standalone \
    --nproc_per_node="$NPROC" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m scripts.chat_sft \
        --base-ckpt "checkpoints/${RUN}/latest.pt" \
        --data-dir data/sft \
        --device-batch-size 1 \
        --grad-accum 8 \
        --max-steps 3000 \
        --run-name "${RUN}_sft"
fi

# ===========================================================================
# Stage 5: RL (GRPO on MBPP, executable reward)
#
# RL 通常一张卡就能跑 (rollout 采样为主)，8B 模型除外，仍建议 FSDP。
# 如果 chat_rl.py 暂未加 FSDP，可先缩小到 d24 验证 pipeline。
# ===========================================================================
if [ "$SKIP_TO" -le 5 ]; then
echo "==> [5/5] RL — GRPO on MBPP"
python -m scripts.chat_rl \
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
echo "  tensorboard --logdir runs/tb"
echo ""
echo "Chat:"
echo "  python -m scripts.chat_cli --ckpt checkpoints/${RUN}_rl/latest.pt"
echo ""
echo "Skip to a specific stage on re-run:"
echo "  SKIP_TO=4 bash runs/train_a800_x8.sh"
