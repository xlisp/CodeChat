# 在 SWE-bench 上评测 CodeChat

## 什么是 SWE-bench

[SWE-bench](https://www.swebench.com/) 是目前最严格的"真实世界 Python 代码能力"评测集。每个样本是从 12 个大型开源 Python 仓库（django, sympy, scikit-learn, matplotlib, flask, requests, ...）真实 issue 中抽取的：

- 输入：issue 描述 + 仓库在 bug 提交前的 base_commit
- 输出：一个统一格式的 `git diff` 补丁
- 打分：官方 harness 在 Docker 里 `git checkout base_commit`，`git apply` 你的补丁，然后跑对应的 `FAIL_TO_PASS` + `PASS_TO_PASS` 测试。全部通过才算 **resolved**

有三个常用切片：

| 切片 | 样本数 | 难度 | 说明 |
|---|---|---|---|
| SWE-bench **Lite** | 300 | 相对容易 | 单文件修改为主，适合首次跑通 |
| SWE-bench **Verified** | 500 | 中等 | OpenAI 人工核验过题目质量 |
| SWE-bench **Full** | 2294 | 困难 | 完整原始集合 |

CodeChat 默认跑 **Lite**，因为 2B 规模的小模型在 Verified/Full 上几乎只能拿 0。

## 对 2B 模型要有现实预期

| 模型 | SWE-bench Lite resolved |
|---|---|
| GPT-4 (直接 prompt) | ~2–4% |
| Claude 3.5 Sonnet (agent) | ~40–50% |
| DeepSeek-Coder 33B | ~10–20% (配合 agent scaffolding) |
| **一个从零训练的 2B CodeChat** | **预计 0%–1%** |

为什么这么低？

1. **规模太小**：SWE-bench 里一个 issue 往往要读几千行代码才能定位 bug，2B 模型的世界知识和长上下文都不够
2. **训练分布不匹配**：我们的 SFT/RL 是「给题→写函数」，SWE-bench 是「读 issue → 定位仓库某文件某行 → 改 diff」，这是两种完全不同的任务
3. **没有 agent 脚手架**：SOTA 成绩都依赖 ReAct / SWE-agent 之类的工具调用循环（grep、读文件、跑测试），CodeChat 当前是一次性生成

**所以 SWE-bench 对 CodeChat 的正确用法是：拿它当一个"冷静剂"和回归指标**，用来观察每次训练改动是否让分数从 0/300 变成 1/300、2/300。它不是用来和 GPT-4 比大小的。

## 如何运行

### 第 1 步：让模型生成补丁

```bash
python -m scripts.eval_swebench \
    --ckpt checkpoints/codechat_2b_rl/latest.pt \
    --split lite \
    --out predictions/codechat_2b_rl.jsonl \
    --model-name codechat-2b-rl
```

这一步只用 CodeChat 自己的 GPU，**不需要 Docker**。它会：

1. 从 HuggingFace 拉 `princeton-nlp/SWE-bench_Lite`
2. 对每个 instance，用 `<|user|> ... <|assistant|>` 格式构造 prompt，把 issue 和仓库信息塞进去
3. 模型生成一个 ```` ```diff ... ``` ```` 代码块
4. 抽出 diff，写成 SWE-bench 官方要求的 jsonl：
   ```json
   {"instance_id": "...", "model_name_or_path": "...", "model_patch": "diff --git ..."}
   ```

可选参数：
- `--temperature 0.2 --top-k 50`：代码补丁用低温度更稳
- `--max-new-tokens 1024`：补丁可能较长
- `--limit 20`：只跑前 20 条做 smoke test

### 第 2 步：用官方 harness 打分

打分必须用官方 harness，因为它要在**每个 instance 独立的 Docker 容器**里复现当时的仓库环境并跑测试。不能用我们的本地 `execution.py`（那个只跑单文件 assert）。

```bash
pip install swebench

python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path predictions/codechat_2b_rl.jsonl \
    --max_workers 4 \
    --run_id codechat_2b_rl
```

前置条件：

- 机器上装有 Docker，且当前用户能 `docker run`
- 大约 **120GB 磁盘**用于拉取所有 instance 的镜像（Lite 全跑一次大约 2–4 小时）
- 训练用的 A800 机器如果没有 Docker，可以把 `predictions/*.jsonl` 拷到另一台有 Docker 的 CPU 机器上跑 harness，两边完全解耦

harness 跑完会在当前目录生成 `logs/run_evaluation/codechat_2b_rl/...` 和一份 summary，关键指标是：

```
resolved: X / 300
applied:  Y / 300      # 补丁能 git apply 的数量
```

`applied` 是一个更友好的早期指标——哪怕 resolved 是 0，能把 applied 从 0 拉到 20 就已经是明显进步。

## 把 SWE-bench 分数纳入训练闭环

建议在 `runs/train_a800.sh` 结尾追加一步：

```bash
echo "==> [6/6] SWE-bench Lite eval (patch generation only)"
python -m scripts.eval_swebench \
    --ckpt "checkpoints/${RUN}_rl/latest.pt" \
    --split lite \
    --out "predictions/${RUN}_rl.jsonl" \
    --model-name "${RUN}-rl" \
    --limit 50
```

只跑 50 条是为了快速反馈（几分钟），然后在另一台机器上 Docker harness 拿完整 300 条的 resolved 分数作为长周期指标。

## 进一步提升 SWE-bench 分数的方向

纯从 loss/RL 层面能做的已经不多，真正的杠杆在于**任务形式**：

1. **切到 agent 范式**：给模型加 `grep` / `read_file` / `run_tests` / `write_file` 工具（参考 SWE-agent），多轮交互。这会把任务从"一次性生成补丁"变成"探索 + 定位 + 修复"，对小模型友好很多
2. **用 SWE-bench 训练集做 SFT**：`princeton-nlp/SWE-bench` 有 `train` split，可以把 `(issue, gold_patch)` 对加入到 `scripts/prepare_sft.py` 的数据混合里，让模型直接见过 diff 格式
3. **合成 diff 数据**：从任意 GitHub commit 生成 `(before, commit_msg, diff)` 三元组作为额外 SFT / RL 语料
4. **GRPO on SWE-bench train**：在我们的 RL 阶段之外，加一个以 SWE-bench train 为环境的 RL 阶段，奖励是 `patch apply + tests pass`（需要在 RL loop 里集成 Docker，非常重）

## 参考

- Jimenez et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*
- Yang et al. (2024). *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*
- 官方仓库：https://github.com/princeton-nlp/SWE-bench
- 排行榜：https://www.swebench.com/
