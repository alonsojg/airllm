#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: scripts/setup_fork_remote.sh <github-username>"
  exit 1
fi

USER_NAME="$1"
FORK_URL="https://github.com/${USER_NAME}/airllm.git"

if git remote get-url upstream >/dev/null 2>&1; then
  :
else
  git remote rename origin upstream
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "${FORK_URL}"
else
  git remote add origin "${FORK_URL}"
fi

echo "Remotes configured:"
git remote -v

echo "Push your branch with:"
echo "git push -u origin \"$(git branch --show-current)\""
