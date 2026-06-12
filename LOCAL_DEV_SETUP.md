# AtmoLLM Local Dev Setup

This setup is tested in this workspace on Linux.

## Fast path

```bash
scripts/dev_bootstrap.sh
scripts/dev_smoke_test.sh
```

What this does:

- Creates `.venv` if missing.
- Installs editable AtmoLLM and dev dependencies.
- Applies compatibility pins required by current AirLLM-compatible imports.
- Runs import + unit-test smoke checks.

## 1) Create and use venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 2) Install AtmoLLM editable package

```bash
python -m pip install -e ./air_llm
```

## 3) Install extras used by tests and bug work

```bash
python -m pip install evaluate scikit-learn sentencepiece wandb peft pytest
```

## 4) Pin compatibility for the import path

AtmoLLM keeps compatibility with the original AirLLM import path. The current code also uses `optimum.bettertransformer`, which requires `transformers<4.49`.

```bash
python -m pip install "optimum<2" "transformers<4.49"
```

## 5) Smoke test

```bash
python -c "import atmollm; from atmollm import AutoModel; import torch; print('ok', torch.__version__)"
```

## 6) Lightweight unit test

```bash
python -m pytest air_llm/tests/test_automodel.py -q
```

## Notes

- `requirements.txt` contains older pins (for example `scikit-learn==1.2.2`) that do not build on Python 3.13.
- For Python 3.13, use newer compatible versions as installed above.
- If you want strict reproduction of old dependency pins, use Python 3.10 or 3.11.
- `bitsandbytes` may warn if CUDA runtime libraries are not available in your environment.

## Fork workflow for your custom version

After creating your own GitHub fork, set remotes like this:

```bash
scripts/setup_fork_remote.sh <your-user>
git push -u origin "$(git branch --show-current)"
```

Sync upstream changes later with:

```bash
git fetch upstream
git rebase upstream/main
```
