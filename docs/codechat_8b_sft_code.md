## runs/train_a800_x8_v3_funcall.sh（Stage 2，code SFT 续训）
```
  产出链路

  checkpoints/codechat_8b/latest.pt          (预训练，train_a800_x8.sh)
          ↓  通用 SFT
  checkpoints/codechat_8b_sft/latest.pt      (通用 SFT，train_a800_x8.sh)
          ↓  代码数据续训  ← 就在这步产出 sft_code
  checkpoints/codechat_8b_sft_code/latest.pt (train_a800_x8_v3_funcall.sh Stage 2)

  背景（v3 的初衷）

  v3 脚本的 Stage 2 专门为救代码能力而加 —— v2 发现 codechat_8b_sft 在 MBPP 上 parseable 只有 1.75%、pass@8=0，RL 根本启动不了。于是在 codechat_8b_sft
  之上用 MBPP(非 train 切分) + Codeforces-Python + the-stack-smol 三个源再 SFT 2000 步，产出
  codechat_8b_sft_code/latest.pt（runs/train_a800_x8_v3_funcall.sh:20，runs/train_a800_x8_v3_funcall.sh:56）。

  具体命令（v3 Stage 2）

  大致等价于：

  torchrun --nproc_per_node=8 -m scripts.chat_sft \
      --base-ckpt checkpoints/codechat_8b_sft/latest.pt \
      --data-dir data/sft_code \
      --device-batch-size 1 --grad-accum 8 \
      --max-steps 2000 --lr 2e-5 \
      --run-name codechat_8b_sft_code

  数据由 scripts/prepare_sft_code.py 生成到 data/sft_code/train.jsonl（runs/train_a800_x8_v3_funcall.sh:19）。

  v6 里的角色

  v6 不直接用 codechat_8b_sft_code 作为 BASE_CKPT —— 而是用更早的 codechat_8b_sft（通用 SFT），把 code 数据和 funcall 数据一起混入 joint
  SFT。这样两种能力从同一起点同时学起，不会出现"先学代码再学 funcall 互相覆盖"的问题。

```

