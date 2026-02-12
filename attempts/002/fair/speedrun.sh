#!/bin/bash
set -e
# Attempt: 002-fair (optimized)
# d26 + fp8 + 1M batch size + ratio 8.25
# All 30-item experiment plan optimizations applied.
# See attempts/002/h200-opts/NOTES.md for full details and source citations.

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
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
# Source: nanochat PR #128 https://github.com/karpathy/nanochat/pull/128 + arXiv:2411.09009 — cut-cross-entropy for fused CE (item 6)
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
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Source: NOTES.md item #2 — reduce eval overhead for faster speedrun (item 4)
# Source: items 6+7 — fused CE unlocks memory for larger device batch size 32 (item 8)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=26 \
    --target-param-data-ratio=8.25 \
    --fp8 \
    --compile-mode=$COMPILE_MODE \
    --run=$WANDB_RUN \
    --eval-every=1000 \
    --sample-every=-1 \
    --core-metric-every=999999 \
    --device-batch-size=32

torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --device-batch-size=16 \
    --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
python -m nanochat.report generate

echo "$(date '+%Y-%m-%d %H:%M:%S') - Run completed (compile_mode=$COMPILE_MODE)" >> "$ATTEMPT_DIR/results.txt"

# -----------------------------------------------------------------------------
# Source: Chinchilla arXiv:2203.15556 + Hagele et al. arXiv:2405.18392 — sweep lower param/data ratios (item 29)
# Uncomment to run ratio sweep after the main run:
# for RATIO in 8.0 7.75 7.5; do
#     echo "--- Ratio sweep: $RATIO ---"
#     torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
#         --depth=26 \
#         --target-param-data-ratio=$RATIO \
#         --fp8 \
#         --compile-mode=$COMPILE_MODE \
#         --device-batch-size=32 \
#         --eval-every=1000 \
#         --sample-every=-1 \
#         --core-metric-every=999999 \
#         --model-tag=d26-ratio${RATIO} \
#         --run=${WANDB_RUN}-ratio${RATIO}
# done
