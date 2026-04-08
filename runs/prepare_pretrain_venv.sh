#!/usr/bin/env bash
# Create an isolated venv that pins datasets<4.0 (which still supports
# loading scripts) and run scripts/prepare_pretrain.py inside it.
# Only the data download/tokenize step uses this venv; training keeps
# using the system Python.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv_data"
OUT_DIR="${OUT_DIR:-data/pretrain}"
MAX_SHARDS="${MAX_SHARDS:-32}"

# can not use proxy for it
if [ ! -d "${VENV_DIR}" ]; then
    echo "==> creating venv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --upgrade pip
    "${VENV_DIR}/bin/pip" install \
        "datasets<4.0" \
        "huggingface_hub<0.24" \
        "fsspec<=2024.5.0" \
        numpy \
        tqdm \
        tiktoken \
        pyarrow \
        requests
fi

# use proxy
echo "==> running prepare_pretrain.py inside ${VENV_DIR}"
cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}" "${VENV_DIR}/bin/python" -m scripts.prepare_pretrain \
    --out-dir "${OUT_DIR}" \
    --max-shards "${MAX_SHARDS}"
