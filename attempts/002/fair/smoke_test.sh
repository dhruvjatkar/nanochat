#!/bin/bash
set -e
# Smoke test: validate attempt 002 setup (venv, deps, tokenizer, imports).
# Run with 1 GPU and short time to verify readiness for full speedrun.

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1
# Use scratch for uv cache when running from /scratch to avoid home quota
_REPO_ROOT_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[[ "$_REPO_ROOT_ABS" == /scratch/* ]] && export UV_CACHE_DIR="$(dirname "$_REPO_ROOT_ABS")/.cache/uv"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
[[ "$_REPO_ROOT_ABS" == /scratch/* ]] && export NANOCHAT_BASE_DIR="$(dirname "$_REPO_ROOT_ABS")/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
[[ -n "$UV_CACHE_DIR" ]] && mkdir -p "$UV_CACHE_DIR"

REPO_ROOT="$_REPO_ROOT_ABS"
ATTEMPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [ -d "$ATTEMPT_DIR/scripts" ] || [ -d "$ATTEMPT_DIR/nanochat" ]; then
    export PYTHONPATH="$ATTEMPT_DIR:$PYTHONPATH"
    echo "Using local overrides from: $ATTEMPT_DIR"
fi

# Venv + deps
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
uv pip install cut-cross-entropy

# Quick import check (attempt 002 gpt)
python -c "from nanochat.gpt import GPT, GPTConfig; print('Import OK')"

# Report reset
python -m nanochat.report reset

# Tokenizer pipeline (dataset + train + eval)
python -m nanochat.dataset -n 8
python -m scripts.tok_train
python -m scripts.tok_eval

echo "Smoke test passed - ready for full run"
