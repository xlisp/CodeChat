## python env

```bash
rm -rf .venv_train

python3 -m venv .venv_train

source .venv_train/bin/activate

which python

pip install -r requirements.txt

bash runs/train_a800_x8.sh

```

## download dataset by proxy: `pip install ipython`

```py
from datasets import load_dataset

load_dataset("google-research-datasets/mbpp", "sanitized", split="train")

load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

load_dataset("sahil2801/CodeAlpaca-20k", split="train")
```

## ~/.cache/huggingface/datasets => 31M

```
google-research-datasets___mbpp  iamtarun___python_code_instructions_18k_alpaca  sahil2801___code_alpaca-20k
```
