#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install -e "${ROOT_DIR}/air_llm"
"${PYTHON_BIN}" -m pip install evaluate scikit-learn sentencepiece wandb peft pytest
"${PYTHON_BIN}" -m pip install "optimum<2" "transformers<4.49"

echo "Bootstrap complete."
echo "Activate with: source .venv/bin/activate"
