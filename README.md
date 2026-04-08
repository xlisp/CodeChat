# CodeChat

CodeChat 是一个参考 [nanochat](https://github.com/karpathy/nanochat) 实现的极简 Python 代码问答模型训练框架。目标是在**单卡 NVIDIA A800-SXM4-80GB** 上，从公开的 Python 代码/指令数据集出发，端到端完成「预训练 → 指令微调（SFT）→ 可执行奖励强化学习（RL）→ 对话推理」的完整流程，产出一个可以回答 Python 代码问题的小模型。

## 为什么要有 RL 阶段？

对代码问答来说 RL 非常自然：代码是否正确**可以直接被执行器验证**，无需额外训练奖励模型，也无需人工偏好标注。CodeChat 的 RL 阶段采用 GRPO 风格：

- 从 [`mbpp`](https://huggingface.co/datasets/mbpp) (sanitized) 中抽题，每题用当前策略采样 G 条回答
- 每条回答抽出代码，放到子进程中对 `test_list` 里的 `assert` 跑一遍，通过比例作为 reward（0~1）
- 组内归一化得到 advantage，再用「PG loss + KL 到参考模型」更新策略
- 参考模型就是 SFT checkpoint 本身，冻结不动，防止策略漂走

这个阶段可以显著提升模型在实际 Python 题目上的一次通过率，尤其是修好 SFT 后常见的「语法对但跑不通」问题。

## 特点

- **极简可读**：核心代码集中在 `codechat/` 与 `scripts/` 下，无庞大配置对象，易 fork 易 hack
- **单卡 A800 友好**：默认超参已按 80GB 显存 + bf16 调校，无需多机多卡
- **面向 Python 代码问答**：预训练语料与 SFT 语料均选自公开的 Python 代码数据集
- **nanochat 风格**：模型是纯 PyTorch 实现的 GPT，精度通过全局 `COMPUTE_DTYPE=bfloat16` 显式控制，不使用 `autocast`

## 环境要求

| 项目 | 版本 |
|---|---|
| GPU | NVIDIA A800-SXM4-80GB |
| CUDA | 12.8 |
| PyTorch | 2.7.1+cu128 |
| Python | 3.11 |
| 计算精度 | bfloat16 (A800 原生支持) |

## 安装

```bash
cd CodeChat
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 使用的公开数据集

全部通过 HuggingFace `datasets` 自动下载，无需手工准备：

| 阶段 | 数据集 | 说明 |
|---|---|---|
| 预训练 | [`codeparrot/github-code-clean`](https://huggingface.co/datasets/codeparrot/github-code-clean) (Python 子集) | 清洗后的 GitHub Python 源码 |
| SFT | [`iamtarun/python_code_instructions_18k_alpaca`](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) | 1.8w 条 Python 指令问答对 |
| SFT 混合 | [`sahil2801/CodeAlpaca-20k`](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) | 2w 条通用代码指令 |
| RL | [`mbpp`](https://huggingface.co/datasets/mbpp) (sanitized) | 带 `test_list` 的 Python 题目，可执行验证 |

分词器直接复用 GPT-2 BPE（`tiktoken` 的 `gpt2` 编码），在代码场景下压缩率足够，无需另行训练。

## 端到端一键训练

```bash
bash runs/train_a800.sh
```

该脚本依次执行：

1. `scripts/prepare_pretrain.py`   下载并分片预训练语料（约 5–10GB，可通过 `--max-shards` 控制）
2. `scripts/base_train.py`          单卡预训练一个 `depth=20` 的 GPT（约 560M 参数，bf16 下峰值显存 ~55GB）
3. `scripts/prepare_sft.py`         拉取并格式化 SFT 指令数据
4. `scripts/chat_sft.py`            在预训练 checkpoint 上做指令微调
5. `scripts/chat_rl.py`             在 SFT checkpoint 上做可执行奖励 RL（GRPO on MBPP）
6. `scripts/chat_cli.py`            启动命令行对话，测试你的 CodeChat

## 模型规格（默认 `--preset=2b`）

| 参数 | 值 |
|---|---|
| 层数 | 32 |
| 隐藏维度 | 2560 |
| 注意力头数 | 20 (head_dim=128) |
| 上下文长度 | 2048 |
| 参数量 | **~2.1B** |
| 词表大小 | 50257 (GPT-2 BPE，tied embedding) |
| 优化器 | AdamW + cosine LR |
| 训练精度 | bfloat16 |
| 激活检查点 | 开启（fit 80GB 必需） |

内置预设（`--preset`）：

| 预设 | 层数 | 宽度 | 头数 | 参数量 | 备注 |
|---|---|---|---|---|---|
| `d20` | 20 | 1280 | 10 | ~0.4B | 快速迭代 |
| `d24` | 24 | 1792 | 14 | ~0.9B | 中等规模 |
| `2b`  | 32 | 2560 | 20 | ~2.1B | **默认**，A800 80GB |
| `3b`  | 32 | 3072 | 24 | ~3.0B | 80GB 上非常紧，需进一步减小 batch |

### 2B 模型在 A800 80GB 上的显存预算

| 项目 | 大小（bf16 / fp32 混合） |
|---|---|
| 模型权重 (bf16) | ~4.2 GB |
| 梯度 (bf16) | ~4.2 GB |
| AdamW 状态 (fp32 m+v) | ~17 GB |
| 参数主副本 (fp32，用于优化器精度) | ~8.5 GB |
| 激活（bs=2, seq=2048, grad_ckpt 开启） | ~25–35 GB |
| **合计峰值** | **约 60–70 GB** |

所以默认 `--device-batch-size=2`, `--grad-accum=16`（全局 batch = 32 条 × 2048 token ≈ 65K token/step），在 A800 80GB 下可以安全跑起来。若仍接近 OOM，先把 `--device-batch-size` 降到 1、`--grad-accum` 升到 32；仍不行再切回 `--preset=d24`。

## 单独运行各阶段

```bash
# 1. 准备预训练数据（Python 源码，自动分片成 .bin）
python -m scripts.prepare_pretrain --out-dir data/pretrain --max-shards 8

# 2. 预训练（2B 默认）
python -m scripts.base_train \
    --data-dir data/pretrain \
    --preset 2b \
    --device-batch-size 2 \
    --grad-accum 16 \
    --max-steps 30000 \
    --run codechat_2b

# 3. 准备 SFT 数据
python -m scripts.prepare_sft --out-dir data/sft

# 4. 指令微调
python -m scripts.chat_sft \
    --base-ckpt checkpoints/codechat_2b/latest.pt \
    --data-dir data/sft \
    --max-steps 3000

# 5. 可执行奖励 RL（GRPO on MBPP）
python -m scripts.chat_rl \
    --sft-ckpt checkpoints/codechat_2b_sft/latest.pt \
    --max-steps 1000 \
    --group-size 4

# 6. 对话测试
python -m scripts.chat_cli --ckpt checkpoints/codechat_2b_rl/latest.pt
```

> ⚠️ RL 阶段会在子进程里**真的执行模型生成的 Python 代码**以计算奖励。`codechat/execution.py` 只做了 timeout 保护，并不是真正的安全沙箱，请只在可信的训练机器上运行，不要把它暴露给不可信输入。

## 目录结构

```
CodeChat/
├── README.md
├── requirements.txt
├── pyproject.toml
├── codechat/
│   ├── __init__.py
│   ├── common.py           # COMPUTE_DTYPE / device 等全局工具
│   ├── gpt.py              # GPT Transformer 模型定义
│   ├── tokenizer.py        # tiktoken GPT-2 BPE 包装
│   ├── dataloader.py       # 分布式/单卡数据加载器
│   ├── optim.py            # AdamW + cosine LR
│   ├── execution.py        # RL 奖励：子进程执行代码 + assert 判分
│   └── checkpoint.py       # 保存/加载 checkpoint
├── scripts/
│   ├── prepare_pretrain.py # 下载并分片 Python 预训练语料
│   ├── prepare_sft.py      # 下载并格式化 SFT 指令数据
│   ├── base_train.py       # 预训练入口
│   ├── chat_sft.py         # SFT 入口
│   ├── chat_rl.py          # GRPO 风格 RL 入口（MBPP 可执行奖励）
│   └── chat_cli.py         # 命令行对话入口
└── runs/
    └── train_a800.sh       # 单卡 A800 端到端一键脚本
```

## 致谢

- 本项目的模型与训练循环结构大量参考自 [Andrej Karpathy 的 nanochat](https://github.com/karpathy/nanochat)
- 数据集来自 HuggingFace 社区：`codeparrot`、`iamtarun`、`sahil2801`

## License

MIT
