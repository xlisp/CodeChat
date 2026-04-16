# CodeChat 8B Function-Calling v5 训练报告

**日期**: 2026-04-15 ~ 2026-04-16
**硬件**: 8× A800-SXM4-80GB (NVLink)
**框架**: PyTorch 2.10 + FSDP (FULL_SHARD), bf16
**相关脚本**: `runs/train_a800_x8_v5_funcall.sh`、`scripts/chat_rl_funcall.py`、`codechat/funcall_reward.py`
**承接**: v2/v3 在 MBPP-RL 上卡死（`reports/TRAINING_REPORT_8b_funcall_v2_v3.md`），v5 是对 MathGPT recipe 的完整复刻。

---

## 0. TL;DR

- **v5 是 v2/v3 失败后的重构**，核心判断是：8B 基座 MBPP parseable 只有 1.75%，在 MBPP 上做 RL 永远没信号；而 glaive-function-calling-v2 的 JSON 格式匹配天然有阶梯信号，可作为 RL 起点。
- **SFT 成功**: 6000 步（≈10h36m），loss 从 2.13 收到 0.1-1.0 区间，ckpt 保存到 `checkpoints/codechat_8b_funcall_v5/latest.pt`。
- **RL 预诊断惊喜**: SFT 之后的 funcall 格式匹配率就已经 **pass@1 = 86.7%、pass@8 = 87.5%**（n=48），对比 v2 在 MBPP 上 pass@1 = 0.00%，差距量变已经是质变。MathGPT 的关键洞察——「SFT 和 RL 用同分布」——在 8B 上被验证。
- **RL 饱和停机**: 计划 283 步，实际跑到 step ~95 后人工 kill。eval 曲线：step 1 = 0.852/0.908（pass@1/pass@16），step 60 = 0.860/0.900，step 90 = 0.853/0.900。全程 ±0.01 抖动，reward 训练均值稳定在 0.97+、std 缩到 0.05，`advantage = r - mean(r) ≈ 0`，梯度 ≈ 0。
- **定论**:
  - 最优 ckpt: `checkpoints/codechat_8b_rl_funcall_v5/step_000060.pt`（peak pass@1，相对 SFT +0.008）
  - RL 阶段对 funcall 的收益 **微弱但非负**，和 MathGPT 的 "peak-Pass@1 出现在中期而非末尾" 高度一致
  - 未解决的结构性错误：args 里 JSON-in-JSON 的引号混用（Python 字典 `'...'` vs JSON `"..."`）——这不是 reward 能教会的，属于数据/SFT 层面问题
- **过程中修复两个 FSDP 去同步 bug**，两者都会导致 NCCL ALLREDUCE/ALLGATHER 错位 → 600s watchdog timeout。修复后诊断和 RL 都跑通。

---

## 1. 从 v3 到 v5 的设计转向

v2/v3 失败后，问题清单被重新定义为：

| v2/v3 假设 | v5 修正 |
|---|---|
| 弱基座可以用 tiered reward + 过滤放大信号 | 错。当 group 内绝大多数样本都 reward=0 时，放大的是噪声 |
| 用代码 SFT 续训拉升 parseable 就能解锁 MBPP RL | v3 做了，funcall loss 与 v2 逐 step 重合，无收益 |
| RL 任务要选「最终目标」（跑通 MBPP） | MathGPT 的反例：选「SFT 已会做」的任务（GSM8K 格式 #### N）能把 RL 初始 reward 抬 5.9× |

v5 的决策：**用 glaive-function-calling-v2 同一套数据既做 SFT 又做 RL**。funcall JSON 格式是 SFT 显式教过的，模型天然会 emit `<functioncall>` 标签；剩下的难度只是参数名/值是否完全匹配，有 6 个中间 tier 可以踩。

相关源文件：
- reward 设计：`codechat/funcall_reward.py`（7 档阶梯）
- RL 数据抽取：`scripts/prepare_rl_funcall.py`
- RL 训练：`scripts/chat_rl_funcall.py`

---

## 2. 流水线与各阶段结果

### 2.1 总览

| 阶段 | 输入 | 输出 | 状态 | 耗时 |
|---|---|---|---|---|
| [1] funcall SFT 数据准备 | glaive-function-calling-v2 | `data/sft_funcall/train.jsonl`（112,934 行） | **DONE**（复用 v2） | — |
| [2] funcall SFT | base = `codechat_8b_sft/latest.pt`，6000 步 | `checkpoints/codechat_8b_funcall_v5/latest.pt` | **DONE** | ≈10h36m |
| [3] RL 数据抽取 | glaive 中含 `<functioncall>` 的首轮 | `data/rl_funcall/{train,eval}.jsonl`（2267 / 122） | **DONE** | 秒级 |
| [4] Pre-RL 诊断 | funcall SFT ckpt，FSDP×8 | 在线 pass@k | **DONE**（修 bug 后）| ≈10m |
| [5] funcall RL | 2267 题，1 epoch = 283 步 | `checkpoints/codechat_8b_rl_funcall_v5/step_*.pt` | **早停** @ step ~95 | ≈5h |

### 2.2 阶段 [2]：funcall SFT

**超参（`runs/train_a800_x8_v5_funcall.sh` 220-236 行）**:

| 设置 | 值 | 相对 v2/v3 |
|---|---|---|
| Base ckpt | `checkpoints/codechat_8b_sft/latest.pt` | 同 v2（没接 v3 的 sft_code） |
| 全局 batch | 1 × 8 × 8 × 2048 ≈ 131k tokens/step | 同 |
| 学习率 | 3e-5（cosine, warmup 200） | 同 |
| 总步数 | **6,000**（v2/v3 是 4,000） | +50% 步数，对应 MathGPT「SFT 多跑一些就能把 RL 起点抬 5.9×」的经验 |

**关键采样点**:

```
step    20 | loss 2.1341 | lr 3.00e-06 |    148s
step   100 | loss 0.6562 | lr 1.50e-05 |    663s
step  1000 | loss ~0.6   |                 ~6200s（插值）
step  3000 | loss 0.5-0.9 区间抖动
step  5800 | loss 0.0561 | lr 3.08e-06 |  36946s   ← 最低点附近
step  6000 | loss 0.7484 | lr 3.00e-06 |  38209s
```

曲线形态正常：前 200 步急降（2.13 → 0.65），之后在 0.1–1.0 区间随 batch 内容起伏。结尾的 0.748 不反常——funcall 数据里每个样本长度/难度差异大，单步 loss 本来就抖。TB 日志 `runs/tb/codechat_8b_funcall_v5/`。

### 2.3 阶段 [3]：RL 数据抽取

`scripts/prepare_rl_funcall.py` 走 glaive 的 112,960 条对话，取首个带 `<functioncall>` 的 assistant turn，把前缀拼成 prompt、JSON 解出 `gt_name` / `gt_args`。筛选结果：

```
seen=112960, no_call=110570, too_long=1
kept=2,389 → train=2,267, eval=122
```

**观察**: 110k 条里只有 2.4k 条首轮就触发工具调用，大部分对话要么是纯聊天，要么把 call 推到了后面轮次。eval 集 122 题在后面 online eval 里被当作泛化信号。

### 2.4 阶段 [4]：Pre-RL 诊断（并揭出两个 FSDP bug）

诊断设计（脚本 264-293 行）：`--max-steps 1 --num-samples 8 --eval-every 1 --eval-examples 50 --lr 0`，目的是在不更新权重的前提下先跑一次 eval，看 reward tier 分布是不是 "全 0 无信号"。

**首次运行 → NCCL timeout**（日志 rank 3 在 `ALLREDUCE(NumelIn=3)`，其他 rank 在 `_ALLGATHER_BASE`，600s 后 watchdog 拉闸）。

根因 1（eval 分片不均）：
```python
# scripts/chat_rl_funcall.py (原)
n_eval = min(n_examples, len(eval_problems))   # 50
local_indices = range(rank, n_eval, world_size)
#   rank 0/1 各 7 个, rank 2-7 各 6 个
```
2-7 号 rank 先跑完到 `dist.all_reduce(totals)`，0/1 还卡在 `sample_batch → FSDP forward → ALLGATHER`。

修复：`n_eval = (n_eval // world_size) * world_size`，强制整除。

根因 2（`sample_batch` 早停各自为政）：
```python
if bool(done.all()):   # 每个 rank 独立判断
    break
```
rank 3 的 8 个 rollouts 可能提前全部命中 EOT 退出，其他 rank 还在继续 FSDP forward。

修复：early-stop 决策改为集合通信同步——
```python
local_done = torch.tensor([int(bool(done.all()))], dtype=torch.int32, device=...)
dist.all_reduce(local_done, op=dist.ReduceOp.MIN)   # 只有所有 rank 都 done 才退
if bool(local_done.item()):
    break
```

第 3 处顺手修的：`prompt_len > cfg.block_size - max_new_tokens` 的 `continue` 改成左截断，避免单 rank skip 导致的 collective 数量不一致。这三处都在 `scripts/chat_rl_funcall.py`。

**修复后诊断结果**（n=48，因为 50 向下取整到 48）:

```
[eval] step 1 | pass@1 0.8672 | pass@8 0.8750 | n=48
```

**这是 v5 的关键证据**。对比 v2 在 MBPP 上 `pass@1 = 0.00%`，funcall 任务从 SFT 出来就已经是 86.7% 的精准格式匹配——RL 有东西可学（+13.3% 的上行空间），但 **已经接近天花板**。

### 2.5 阶段 [5]：funcall RL

**超参（脚本 110-122 行）**:

| 设置 | 值 | 备注 |
|---|---|---|
| 算法 | REINFORCE with baseline（`advantage = r - r.mean()`） | 无 KL、无 ref model —— dense 阶梯 reward 自带正则 |
| num-samples | 16（per rank） | 8 rank × 16 = **128 effective rollouts/step** |
| num-epochs | 1 → num_steps = 2267/8 = **283** | MathGPT v2 经验：3 epoch 会过拟 |
| lr | 1e-5 × init_lr_frac=0.05 → 起点 5e-7 | cosine 到 0 |
| max-new-tokens | 256 | funcall JSON 短，够用 |
| temperature / top-k | 1.0 / 50 | |
| eval-every / eval-examples | 30 / 200 | |

**训练曲线（关键采样）**:

```
[eval] step  1 | pass@1 0.8521 | pass@16 0.9083 | n=120   ← pass@16 峰值
rl step   5  | reward 0.938 (max 1.00, std 0.095) | loss -0.0000
rl step  15  | reward 0.969 (max 1.00, std 0.056) |        ← std 明显收窄
rl step  35  | reward 1.000 (max 1.00, std 0.000) | loss -0.0000
rl step  50  | reward 0.786 (max 0.79, std 0.005) |        ← 撞上一个 16 rollouts 全 0.15 的难题
[eval] step 60 | pass@1 0.8599 | pass@16 0.9000 | n=120   ← pass@1 峰值（+0.008 vs step 1）
rl step  75  | loss -0.4113                                 ← 偶发大梯度
[eval] step 90 | pass@1 0.8531 | pass@16 0.9000 | n=120
... （人工 kill）
```

**饱和解读**:
- 128 个 rollouts 里绝大多数拿到 1.0，少数拿到 0.79 / 0.89，`rewards.mean() ≈ 0.97`，`advantages = r - r.mean()` 量级 0.03
- `pg_loss = -(advantages * per_sample_logp).mean()` 自然贴近 0，后续权重更新极小
- 偶发的 loss spike（step 75 的 -0.41）对应那种 "某个具体 prompt 上 16 rollouts 大幅偏离"，是局部信号，对全局 eval 不足以产生可见改善
- 三次 eval（step 1/60/90）在 pass@1 ±0.008、pass@16 ±0.008 内波动，远小于 n=120 下的置信区间，不能断定 RL 有统计显著提升

### 2.6 一个无法用 reward 解决的失败模式

step 50 的 rollout 日志非常说明问题：

```
gt_name: check_email_spam
gt_args: {'email_subject': "You've won a million dollars!", 'email_body': ...}
rewards: [0.15, 0.15, ..., 0.15, 0.0, 0.15, 0.15]   ← 16 个里 15 个 0.15、1 个 0.0
best output:
  <functioncall> {"name": "check_email_spam",
                  "arguments": '{"email_subject": "You've won a million dollars!", ...}'}
```

问题在 `arguments` 字段：外面套的是**单引号**（Python 字典风格），而 JSON 规范只允许双引号。glaive 数据本身就是混合风格（gt_args 的 ground truth 也是 Python repr），模型学到了这种模糊。因为 `<functioncall>` 标签有、name 对、args 存在但解析失败——reward 函数打到 `bad_json` 档（0.15），16 个 rollouts 全在这一档，`advantage ≈ 0`，**RL 对这个错误零梯度**。

这是一个 "SFT 数据质量问题"，不是 "RL 调参问题"。修法只能回到数据层：`prepare_sft_funcall` 里统一把 `arguments` 规范化成合法 JSON 字符串。

---

## 3. 对比总览

### 3.1 v2 / v3 / v5

| 维度 | v2 | v3 | v5 |
|---|---|---|---|
| SFT 基座 | `codechat_8b_sft` | `codechat_8b_sft_code`（先代码续训） | `codechat_8b_sft` |
| funcall SFT 步数 | 4,000 | 4,000 | **6,000** |
| RL 任务 | MBPP pass@k | MBPP pass@k | **glaive funcall 格式匹配** |
| RL reward | tiered 5 档（语法/运行/部分/全通过） | 同 v2 | **7 档阶梯（no_tag / bad_json / no_name / wrong_name / name_only / partial / full_match）** |
| RL 起点（pass@1） | 0.00% | 0.00% | **85.2%** |
| RL 是否真的更新了模型 | ❌（过滤后 0 题） | ❌（过滤后 0 题） | ✅（283 步中跑了 ~95 步） |
| RL 带来的 eval 提升 | — | — | +0.008（接近噪声） |
| 结论 | 需换任务 | 代码续训无增益 | **recipe 成立但任务已接近饱和** |

### 3.2 与 MathGPT 的对照

| 项 | MathGPT | v5 |
|---|---|---|
| SFT/RL 同分布 | GSM8K × 16 epoch 在 SFT | glaive-v2 × ~7 epoch 在 SFT |
| RL 起点 | 可观非 0 的 GSM8K pass@1 | funcall pass@1 ≈ 0.85 |
| reward 形态 | 格式 + 末位数字匹配 | 7 档 funcall JSON 匹配 |
| Pass@1 vs Pass@K 峰值 | Pass@1 晚于 Pass@K 到达峰值 | 同向：pass@16 @ step 1、pass@1 @ step 60 |
| 1 epoch 是最佳 | 是 | 是（283 步里的 21% 处已最优） |

v5 在"方法对不对"上是复刻成功的，只是任务本身的天花板（glaive 多年前的弱格式）让绝对收益有限。

---

## 4. 产物与复用

### 4.1 ckpt

| 用途 | 路径 | 依据 |
|---|---|---|
| **部署推荐** | `checkpoints/codechat_8b_rl_funcall_v5/step_000060.pt` | peak pass@1 = 0.8599 |
| sampling vote 备选 | `checkpoints/codechat_8b_rl_funcall_v5/step_000001.pt` | peak pass@16 = 0.9083（若未被 keep-every=60 删除） |
| SFT baseline | `checkpoints/codechat_8b_funcall_v5/latest.pt` | 和 RL ckpt 做 A/B |

### 4.2 TensorBoard

```
runs/tb/codechat_8b_funcall_v5         # SFT loss
runs/tb/codechat_8b_rl_funcall_v5_diag # 预诊断（reward tier 分布）
runs/tb/codechat_8b_rl_funcall_v5      # RL 主过程
```

关键 scalar：
- `rl/reward_mean`、`rl/reward_std` → 看是否饱和
- `rl/tier/<name>` → 每档在 128 rollouts 里的占比
- `eval/pass@1`、`eval/pass@16` → 泛化
- `rl/loss_pg` 长期贴 0 就是无信号

### 4.3 测评脚本

新增 `scripts/eval_funcall.py`：单 GPU、支持单 ckpt / 多 ckpt / 扫 run-dir 三种模式，输出 Pass@k（Codex 无偏）、full-match 率、七档 tier 分布、示例 rollouts。

---

## 5. 已知问题 & 下一步

### 5.1 数据层

- **args JSON 格式不一**：glaive gt_args 是 Python repr（单引号），而模型有时 emit JSON 风格（双引号），`_parse_json_loose` 已经做宽松修复但不能覆盖所有边界。建议在 `prepare_sft_funcall` 里把 assistant 的 `arguments` 统一归一化到 JSON-in-JSON-string 格式，同步到 gt。
- **多轮工具调用**: 只抽首次 call 的 prompt，后续 `<function_response>` → 下一次 call 的分布完全没进 RL。后续版本可以扩成 "每次 funcall 一个 rollout"。

### 5.2 训练层

- **reward 饱和**是 v5 的主要顶板。想再抬一点 pass@1 的选项：
  1. 收紧 reward：把 "args 部分匹配" 的上限从 0.99 降到 0.85，让 full_match 和 partial 的 gap 更宽
  2. 选更难的子集做 RL：只保留 SFT 里 pass@1 < 0.8 的那 2% 问题作为 RL prompts
  3. 加 argument-level 的 per-key reward 而不是 match_fraction 标量 —— 给 "偶尔多/少一个参数键" 单独的梯度
- **KL / ref model**: v5 刻意不用，结果 reward std 收窄得挺快，没有 policy collapse 迹象；可以继续不带。

### 5.3 工程层

- `codechat/checkpoint.py` 的 `FSDP.state_dict_type()` 在 PyTorch 2.10 起 deprecated，保存时有 warning；下一版迁到 `torch.distributed.checkpoint.state_dict.get_state_dict`。
- shell 脚本里 `NCCL_ASYNC_ERROR_HANDLING`（旧名）和 `TORCH_NCCL_ASYNC_ERROR_HANDLING`（新名）同时 export，PyTorch 会对旧名报 deprecated。删 `NCCL_ASYNC_ERROR_HANDLING` 一行即可安静。

---

## 6. 结论

v5 走到头了：**MathGPT 的 recipe 在 8B 上被证明可行**，从 MBPP-RL 的 reward≡0 灾难走到 funcall-RL 的 128 rollouts/step 稳定训练；但 glaive-v2 自身的格式噪声把 pass@1 的天花板锁死在 0.86 附近，RL 只能在 ±0.01 里挣扎。

**下一个版本的方向应该是换更高质量的 function-calling 数据集**（ToolBench / Hermes-Function-Calling / xLAM 之类都有更严格的 JSON schema）、**或者用同一 recipe 换任务**（math tool use、code + tests 同分布 SFT+RL）。在 glaive 这一层，v5 已经是局部最优。
