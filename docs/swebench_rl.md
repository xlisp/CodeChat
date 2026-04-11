# GRPO on SWE-bench Train: Docker-in-the-Loop RL

## 一句话总结

在原有 MBPP RL（"给题 → 写函数 → 跑 assert"）之后，追加一个以 **SWE-bench train** 为环境的 RL 阶段：模型读 GitHub issue → 生成 unified diff 补丁 → 在 Docker 容器里 `git apply` + 跑测试 → 通过率作为 reward。这迫使模型学会一种全新的能力：**在真实仓库上下文中定位并修复 bug**。

## 为什么这能提升模型能力

### 1. 从"写函数"到"改仓库"——任务形式迁移

| | MBPP RL (stage 5) | SWE-bench RL (stage 7) |
|---|---|---|
| 输入 | 一段题目描述 | 真实 GitHub issue + 仓库信息 |
| 输出 | 完整 Python 函数 | unified diff 补丁 |
| 验证 | 本地 subprocess 跑 assert | Docker 里 git apply + pytest/unittest |
| 训练信号 | assert 通过率 (0..1) | patch apply + test pass 率 (0..1) |

MBPP RL 教会模型"给定一个独立问题，写出正确代码"。但真实软件工程中，模型需要理解的是：
- **diff 格式**：`diff --git a/... b/...`、`@@ -L,N +L,N @@`、上下文行、增删行
- **定位能力**：从 issue 描述中判断该改哪个文件的哪些行
- **最小修改原则**：只改必要的地方，不破坏其他测试

这些都不是 MBPP 能训练出来的。SWE-bench RL 直接用"补丁能不能跑通测试"作为 reward，端到端地教模型学这些。

### 2. 分层奖励——从格式到语义的课程学习

我们设计了三级 reward mode 作为课程学习 (curriculum)：

```
syntax (0.3)  →  apply-only (0.3)  →  docker (0.3 + 0.7 * test_pass_rate)
```

| Phase | Reward mode | 模型学到什么 | 速度 |
|---|---|---|---|
| Phase 1 | `syntax` | 输出合法的 unified diff 格式（有 `@@` hunk header） | 秒级/step |
| Phase 2 | `apply-only` | 补丁能在真实仓库上 `git apply`（正确的文件路径、行号、上下文） | ~10s/step |
| Phase 3 | `docker` | 补丁不仅 apply，还能修复 bug（FAIL_TO_PASS 测试从 fail 变 pass） | 1-5min/step |

为什么要分阶段？因为一个 2B 模型一开始连 diff 格式都生成不好。如果直接进入 Phase 3，绝大多数 rollout 的 reward 都是 0（连 apply 都过不了），advantage 方差极大，训练信号几乎为零。分阶段让模型先掌握格式，再学语义。

### 3. Docker-in-the-Loop 提供"真实世界"信号

为什么不能用简单的启发式（比如检查 diff 是否改了正确的文件）替代 Docker？因为：
- **启发式太粗**：改了正确的文件不等于改对了。一个 off-by-one 的 diff 能 apply 但跑不过测试
- **Docker 是 ground truth**：SWE-bench 的 Docker 镜像精确复现了每个 issue 当时的仓库状态、Python 版本、依赖版本。测试通过 = 真的修好了
- **FAIL_TO_PASS 测试是二元反馈**：这是最清晰的 RL 信号——"这行代码改对了吗？测试说：是/否"

代价是速度极慢（每个 Docker eval ~30-300s），但这正是 docs/swebench_eval.md 里说的"非常重"。

### 4. 对 2B 模型的现实预期

即使加了 SWE-bench RL，2B 模型在 SWE-bench Lite 上的 resolved 率仍然很低（预计 0-3%）。但关键的早期指标是：

| 指标 | 没有 SWE-bench RL | 加了 SWE-bench RL（预期） |
|---|---|---|
| 生成有效 diff 格式 | ~5% | ~40-60% |
| patch 能 git apply | ~0% | ~10-20% |
| resolved (测试全通过) | 0/300 | 0-5/300 |

`applied` 比 `resolved` 更早看到进步。即使 resolved 仍然是 0，"能生成 apply 的补丁"本身已经是巨大的能力提升。

## 架构概览

```
┌─────────────────────────────────────────────────────┐
│                  rl_swebench.py                     │
│                                                     │
│  for step in 1..max_steps:                          │
│    instance = random SWE-bench train sample         │
│    prompt = issue_description → chat format         │
│                                                     │
│    ┌─── sample G completions (GPU) ───┐             │
│    │ completion_1 → extract_diff → patch_1          │
│    │ completion_2 → extract_diff → patch_2          │
│    │ ...                                            │
│    │ completion_G → extract_diff → patch_G          │
│    └──────────────────────────────────┘             │
│                     │                               │
│                     ▼                               │
│    ┌─── SWEBenchReward (parallel Docker) ───┐       │
│    │                                        │       │
│    │  ThreadPool(docker_workers)             │       │
│    │    ┌──────────┐  ┌──────────┐          │       │
│    │    │ Docker 1  │  │ Docker 2  │  ...    │       │
│    │    │ apply +   │  │ apply +   │         │       │
│    │    │ run tests │  │ run tests │         │       │
│    │    └────┬─────┘  └────┬─────┘          │       │
│    │         │             │                │       │
│    │    reward_1       reward_2    ...       │       │
│    └────────────────────────────────────────┘       │
│                     │                               │
│                     ▼                               │
│    advantages = normalize(rewards)                  │
│    GRPO update: PG loss + KL penalty                │
│    save checkpoint                                  │
└─────────────────────────────────────────────────────┘
```

## 文件清单

| 文件 | 说明 |
|---|---|
| `codechat/swebench_reward.py` | Docker-based reward 引擎：syntax / apply-only / docker 三种模式 |
| `scripts/rl_swebench.py` | GRPO 训练主循环，SWE-bench train 为环境 |
| `runs/train_a800_v2.sh` | 完整 8 阶段训练脚本（含 SWE-bench RL） |

## 快速开始

### 前置条件

```bash
# 1. Docker (必须)
docker info  # 确认能无 sudo 运行

# 2. swebench 库
pip install swebench

# 3. 磁盘空间：SWE-bench train 的 Docker 镜像约 200-400GB
# 4. 已完成 v1 训练，有 SFT/RL checkpoint
```

### 一键运行（推荐）

```bash
# 完整 v2 训练流水线（含 SWE-bench RL 阶段）
bash runs/train_a800_v2.sh
```

### 分步运行

```bash
# Step 0: 预构建 Docker 镜像（仅需一次，可能需要几小时）
python -m scripts.rl_swebench \
    --sft-ckpt checkpoints/codechat_2b_rl/latest.pt \
    --prepare-images --max-steps 0

# Step 1: syntax warm-up（教 diff 格式，无需 Docker，快）
python -m scripts.rl_swebench \
    --sft-ckpt checkpoints/codechat_2b_rl/latest.pt \
    --reward-mode syntax \
    --max-steps 200 --group-size 4 \
    --run codechat_2b_rl_sweb_syntax

# Step 2: apply-only（教补丁 apply，中速）
python -m scripts.rl_swebench \
    --sft-ckpt checkpoints/codechat_2b_rl_sweb_syntax/latest.pt \
    --reward-mode apply-only \
    --max-steps 300 --group-size 4 --docker-workers 4 \
    --run codechat_2b_rl_sweb_apply

# Step 3: full Docker（教修 bug，慢但信号最强）
python -m scripts.rl_swebench \
    --sft-ckpt checkpoints/codechat_2b_rl_sweb_apply/latest.pt \
    --reward-mode docker \
    --max-steps 500 --group-size 4 --docker-workers 4 \
    --run codechat_2b_rl_sweb

# Step 4: 评测
python -m scripts.eval_swebench \
    --ckpt checkpoints/codechat_2b_rl_sweb/latest.pt \
    --split lite --out predictions/codechat_2b_rl_sweb.jsonl
```

## 关键超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--reward-mode` | `docker` | `syntax` / `apply-only` / `docker` |
| `--group-size` | 4 | 每个 prompt 采样几个 completion（越大方差越小，但 Docker 成本线性增长） |
| `--docker-workers` | 2 | 并行 Docker 容器数（受 CPU 和磁盘 IO 限制） |
| `--docker-timeout` | 300 | 单容器超时（秒）。django 的测试可能很慢，建议 300+ |
| `--max-new-tokens` | 1024 | diff 补丁比函数代码长，需要更多 token |
| `--lr` | 5e-6 | 比 MBPP RL 低（1e-5），因为 SWE-bench 的 reward 更稀疏更噪声 |
| `--kl-coef` | 0.02 | KL penalty 系数，防止 policy 偏离 SFT 太远 |
| `--temperature` | 0.7 | 采样温度。diff 生成比自由文本需要更多确定性，但也不能太低（RL 需要探索） |

## Reward 设计细节

### 奖励分解

```
reward = apply_bonus * I(patch_applies) + test_weight * (tests_passed / tests_total)
```

默认 `apply_bonus=0.3, test_weight=0.7`，这样：

| 情况 | Reward |
|---|---|
| 空 patch / 语法错误 | 0.0 |
| patch 能 apply 但测试没过 | 0.3 |
| patch apply + 50% 测试过 | 0.65 |
| patch apply + 全部测试过 | 1.0 |

### 为什么给 apply 单独加分？

因为对 2B 模型来说，"生成一个能 apply 的补丁"已经很难了。如果只有 0/1 reward（全通过才有分），模型几乎收不到正向信号。`apply_bonus` 让模型在学会写正确 diff 格式之后就开始收到 reward，形成渐进的学习曲线。

### Docker 安全

- 所有 Docker 容器使用 `--network none` 隔离网络
- 容器以 `--rm` 运行，用完即销
- 临时文件在 finally 块中清理

## TensorBoard 监控

```bash
tensorboard --logdir runs/tb
```

SWE-bench RL 阶段的关键指标（前缀 `swebrl/`）：

| 指标 | 含义 | 期望趋势 |
|---|---|---|
| `reward_mean` | 组内平均 reward | 缓慢上升 |
| `patches_nonempty` | 非空 patch 数 (/ group_size) | Phase 1 应快速上升 |
| `patches_applied` | 能 apply 的 patch 数 | Phase 2 应开始上升 |
| `tests_passed_total` | 通过的 FAIL_TO_PASS 测试总数 | Phase 3 偶尔出现非零 |
| `kl` | KL 散度 (policy vs ref) | 不应持续暴涨 |
| `reward_time_s` | Docker 评估耗时 | 了解瓶颈在 GPU 还是 Docker |
| `sample_time_s` | 模型采样耗时 | 通常比 Docker 快很多 |

## 时间和资源估算

以 A800 80GB + 16 核 CPU + SSD 为例：

| Phase | Steps | 每步耗时 | 总时间 | 主要瓶颈 |
|---|---|---|---|---|
| Phase 1 (syntax) | 200 | ~5s | ~17 min | GPU 采样 |
| Phase 2 (apply-only) | 300 | ~15s | ~75 min | Docker 启动 |
| Phase 3 (docker) | 500 | ~2-5 min | ~17-42 h | Docker 跑测试 |
| **合计** | 1000 | — | **~18-44 h** | — |

Phase 3 是绝对瓶颈。可以通过以下方式加速：
- 增大 `--docker-workers`（需要更多 CPU 核心和 IO 带宽）
- 减少 `--group-size` 到 2（牺牲 advantage 估计质量）
- 减少 `--max-steps`（早停）
- 用 `--train-limit N` 只用部分训练实例（减少镜像构建时间）

## 与 v1 训练流水线的关系

```
v1 (train_a800.sh):
  pretrain → SFT → MBPP RL → eval

v2 (train_a800_v2.sh):
  pretrain → SFT → MBPP RL → SWE-bench RL (syntax → apply → docker) → eval
                              ↑ 新增阶段
```

v2 在 v1 的 MBPP RL 之后追加 SWE-bench RL。两个 RL 阶段互补：
- **MBPP RL** 教模型写正确的独立函数（基础代码能力）
- **SWE-bench RL** 教模型在仓库上下文中生成修复补丁（软件工程能力）

KL penalty 确保 SWE-bench RL 不会让模型"忘记" MBPP RL 阶段学到的基础编码能力。

## 局限性与未来方向

1. **极慢**：Docker-in-the-loop 是这个方法最大的成本。未来可以用 learned reward model 替代部分 Docker eval（蒸馏 Docker 的判断）
2. **2B 太小**：SWE-bench 需要读很多上下文代码，2B 模型的世界知识和长上下文能力不够。扩大到 7B+ 效果会更好
3. **单次生成，非 agent**：当前是一次性生成整个 patch。加上 agent scaffolding（grep, read_file, run_tests 工具循环）会大幅提升效果
4. **没有 PASS_TO_PASS 惩罚**：当前只看 FAIL_TO_PASS 是否修复，没有检查是否引入了新的 regression。未来可以加上 PASS_TO_PASS 的负 reward

## 参考文献

- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* — GRPO 算法原始论文
- Jimenez et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*
- Yang et al. (2024). *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*
- 官方仓库：https://github.com/princeton-nlp/SWE-bench
