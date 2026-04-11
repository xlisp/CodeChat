rm -rf .venv_train

python3 -m venv .venv_train

source .venv_train/bin/activate

which python

pip install -r requirements.txt

bash runs/train_a800_x8.sh

# download dataset by proxy:

pip install ipython

```
from datasets import load_dataset

load_dataset("google-research-datasets/mbpp", "sanitized", split="train")

load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

load_dataset("sahil2801/CodeAlpaca-20k", split="train")
```
