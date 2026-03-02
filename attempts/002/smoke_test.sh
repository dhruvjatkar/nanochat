#!/bin/bash
# Attempt: 002 — 1-GPU smoke test (5 iterations, no heavy eval)
# Validates: venv setup, tokenizer, base_train forward/backward, SFT, report generation

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Use scratch for uv/nanochat cache when running from /scratch to avoid home quota
[[ "$REPO_ROOT" == /scratch/* ]] && export UV_CACHE_DIR="$(dirname "$REPO_ROOT")/.cache/uv"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
[[ "$REPO_ROOT" == /scratch/* ]] && export NANOCHAT_BASE_DIR="$(dirname "$REPO_ROOT")/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
[[ -n "$UV_CACHE_DIR" ]] && mkdir -p "$UV_CACHE_DIR"
ATTEMPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# If this attempt has local script or nanochat overrides, prepend to PYTHONPATH
if [ -d "$ATTEMPT_DIR/scripts" ] || [ -d "$ATTEMPT_DIR/nanochat" ]; then
    export PYTHONPATH="$ATTEMPT_DIR:$PYTHONPATH"
    echo "Using local overrides from: $ATTEMPT_DIR"
fi

# -----------------------------------------------------------------------------
# Python venv setup with uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
uv pip install cut-cross-entropy

# -----------------------------------------------------------------------------
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi
COMPILE_MODE=${COMPILE_MODE:-default}

# -----------------------------------------------------------------------------
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining) — 1 GPU, 5 iterations only
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=26 \
    --target-param-data-ratio=8.25 \
    --fp8 \
    --use-teon \
    --compile-mode=$COMPILE_MODE \
    --run=$WANDB_RUN \
    --num-iterations=5 \
    --eval-every=1000 \
    --sample-every=-1 \
    --core-metric-every=999999 \
    --device-batch-size=16

torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT — 1 GPU, 5 iterations only
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
    --device-batch-size=16 \
    --num-iterations=5 \
    --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
python -m nanochat.report generate

echo "$(date '+%Y-%m-%d %H:%M:%S') - Smoke test completed (1-GPU, 5 iterations)" >> "$ATTEMPT_DIR/results.txt"
echo "Smoke test passed!"
