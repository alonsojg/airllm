#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing virtualenv at ${ROOT_DIR}/.venv"
  echo "Run scripts/dev_bootstrap.sh first."
  exit 1
fi

"${PYTHON_BIN}" -c "import airllm; from airllm import AutoModel; import torch; print('airllm import ok'); print('torch', torch.__version__)"
"${PYTHON_BIN}" -m pytest "${ROOT_DIR}/air_llm/tests/test_automodel.py" -q

echo "Smoke test complete."
