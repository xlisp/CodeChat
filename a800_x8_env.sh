rm -rf .venv_train

python3 -m venv .venv_train

source .venv_train/bin/activate

which python

pip install -r requirements.txt

bash runs/train_a800_x8.sh

