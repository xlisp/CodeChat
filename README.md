# CodeChat

CodeChat 是一个参考 [nanochat](https://github.com/karpathy/nanochat) 实现的极简 Python 代码问答模型训练框架。目标是在**单卡 NVIDIA A800-SXM4-80GB** 上，从公开的 Python 代码/指令数据集出发，端到端完成「预训练 → 指令微调（SFT）→ 对话推理」的完整流程，产出一个可以回答 Python 代码问题的小模型。

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
5. `scripts/chat_cli.py`            启动命令行对话，测试你的 CodeChat

## 模型规格（默认 `--depth=20`）

| 参数 | 值 |
|---|---|
| 层数 | 20 |
| 隐藏维度 | 1280 |
| 注意力头数 | 10 |
| 上下文长度 | 2048 |
| 参数量 | ~560M |
| 词表大小 | 50257 (GPT-2 BPE) |
| 优化器 | AdamW + Muon (nanochat 风格) |
| 训练精度 | bfloat16 |

若显存紧张，可降低 `--device-batch-size`（默认 16）或 `--depth`（可选 12 / 16 / 20 / 24）。

## 单独运行各阶段

```bash
# 1. 准备预训练数据（Python 源码，自动分片成 .bin）
python -m scripts.prepare_pretrain --out-dir data/pretrain --max-shards 8

# 2. 预训练
python -m scripts.base_train \
    --data-dir data/pretrain \
    --depth 20 \
    --device-batch-size 16 \
    --max-steps 20000 \
    --run codechat_d20

# 3. 准备 SFT 数据
python -m scripts.prepare_sft --out-dir data/sft

# 4. 指令微调
python -m scripts.chat_sft \
    --base-ckpt checkpoints/codechat_d20/latest.pt \
    --data-dir data/sft \
    --max-steps 3000

# 5. 对话测试
python -m scripts.chat_cli --ckpt checkpoints/codechat_d20_sft/latest.pt
```

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
│   ├── optim.py            # AdamW + Muon 优化器
│   └── checkpoint.py       # 保存/加载 checkpoint
├── scripts/
│   ├── prepare_pretrain.py # 下载并分片 Python 预训练语料
│   ├── prepare_sft.py      # 下载并格式化 SFT 指令数据
│   ├── base_train.py       # 预训练入口
│   ├── chat_sft.py         # SFT 入口
│   └── chat_cli.py         # 命令行对话入口
└── runs/
    └── train_a800.sh       # 单卡 A800 端到端一键脚本
```

## 致谢

- 本项目的模型与训练循环结构大量参考自 [Andrej Karpathy 的 nanochat](https://github.com/karpathy/nanochat)
- 数据集来自 HuggingFace 社区：`codeparrot`、`iamtarun`、`sahil2801`

## License

MIT
