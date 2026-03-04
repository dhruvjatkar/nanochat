#!/bin/bash
# Ablation runner: run a single ablation experiment with d12 (~5 min) defaults.
# Usage:
#   bash ablation.sh --name <name> [--skip-setup] [-- extra_flags...]
#
# Examples:
#   bash ablation.sh --name baseline
#   bash ablation.sh --name lr0.03 -- --matrix-lr=0.03
#   bash ablation.sh --name d16_wide --skip-setup -- --depth=16 --aspect-ratio=80

set -euo pipefail

# =============================================================================
# Parse CLI arguments
# =============================================================================
NAME=""
SKIP_SETUP=0
EXTRA_FLAGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            NAME="$2"
            shift 2
            ;;
        --skip-setup)
            SKIP_SETUP=1
            shift
            ;;
        --)
            shift
            EXTRA_FLAGS=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash ablation.sh --name <name> [--skip-setup] [-- extra_flags...]"
            exit 1
            ;;
    esac
done

if [ -z "$NAME" ]; then
    echo "Error: --name is required"
    echo "Usage: bash ablation.sh --name <name> [--skip-setup] [-- extra_flags...]"
    exit 1
fi

# =============================================================================
# Environment setup (mirrors speedrun.sh / smoke_test.sh)
# =============================================================================
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Use local cache when running outside home to avoid home quota issues
if [[ "$REPO_ROOT" == /scratch/* ]] || [[ "$REPO_ROOT" == /projects/* ]]; then
    export UV_CACHE_DIR="$(dirname "$REPO_ROOT")/.cache/uv"
    export NANOCHAT_BASE_DIR="$(dirname "$REPO_ROOT")/.cache/nanochat"
else
    export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
fi
mkdir -p "$NANOCHAT_BASE_DIR"
[[ -n "${UV_CACHE_DIR:-}" ]] && mkdir -p "$UV_CACHE_DIR"
ATTEMPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# Python venv setup with uv (needs REPO_ROOT as CWD for pyproject.toml)
if [ "$SKIP_SETUP" -eq 0 ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
fi
source .venv/bin/activate

# Switch CWD to attempt dir so local overrides in scripts/ and nanochat/ take
# priority over the base repo (Python's -m puts CWD at sys.path[0]).
# REPO_ROOT in PYTHONPATH provides fallback for non-overridden modules.
if [ -d "$ATTEMPT_DIR/scripts" ] || [ -d "$ATTEMPT_DIR/nanochat" ]; then
    cd "$ATTEMPT_DIR"
    export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
    echo "Using local overrides from: $ATTEMPT_DIR"
fi

# =============================================================================
# Ablation defaults
# =============================================================================
NPROC="${NPROC:-8}"
ABLATION_DEPTH="${ABLATION_DEPTH:-12}"
COMPILE_MODE="${COMPILE_MODE:-default}"

# =============================================================================
# Output directory
# =============================================================================
OUTPUT_DIR="$ATTEMPT_DIR/ablations/$NAME"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/log.txt"
FLAGS_FILE="$OUTPUT_DIR/flags.txt"
RESULT_FILE="$OUTPUT_DIR/result.json"

# Build the full flags list
ALL_FLAGS=(
    --depth="${ABLATION_DEPTH}"
    --run="dummy"
    --model-tag="ablation_${NAME}"
    --compile-mode="${COMPILE_MODE}"
    --target-param-data-ratio="${TARGET_PARAM_DATA_RATIO:-10.5}"
    --core-metric-every=999999
    --sample-every=-1
    --save-every=-1
    "${EXTRA_FLAGS[@]}"
)

# Save flags for reproducibility
{
    echo "# Ablation: $NAME"
    echo "# Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# NPROC: $NPROC"
    echo "# ABLATION_DEPTH: $ABLATION_DEPTH"
    echo "# EXTRA_FLAGS: ${EXTRA_FLAGS[*]:-<none>}"
    echo ""
    echo "torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \\"
    for flag in "${ALL_FLAGS[@]}"; do
        echo "    $flag \\"
    done
    echo ""
} > "$FLAGS_FILE"

echo "============================================================"
echo " Ablation: $NAME"
echo " Depth:    $ABLATION_DEPTH"
echo " GPUs:     $NPROC"
echo " Output:   $OUTPUT_DIR"
echo " Extra:    ${EXTRA_FLAGS[*]:-<none>}"
echo "============================================================"

# =============================================================================
# Ensure dataset is available (tokenizer + data shards)
# =============================================================================
python -m nanochat.dataset -n 8

# =============================================================================
# Run training
# =============================================================================
START_TIME=$(date +%s)

torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
    "${ALL_FLAGS[@]}" \
    2>&1 | tee "$LOG_FILE"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# =============================================================================
# Parse results from log
# =============================================================================
VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$' || echo "")
CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}' || echo "")
NUM_PARAMS=$(grep "total:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | head -1 | tr -d ',' || echo "")

# Find the minimum validation bpb across all eval steps
MIN_VAL_BPB=""
if grep -q "Validation bpb:" "$LOG_FILE"; then
    MIN_VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | grep -oP '[\d.]+$' | sort -n | head -1)
fi

# =============================================================================
# Write result.json
# =============================================================================
cat > "$RESULT_FILE" <<ENDJSON
{
    "name": "$NAME",
    "flags": "${ALL_FLAGS[*]}",
    "min_val_bpb": ${MIN_VAL_BPB:-null},
    "final_val_bpb": ${VAL_BPB:-null},
    "core_score": ${CORE_SCORE:-null},
    "num_params": ${NUM_PARAMS:-null},
    "elapsed_seconds": $ELAPSED,
    "depth": $ABLATION_DEPTH,
    "nproc": $NPROC,
    "extra_flags": "${EXTRA_FLAGS[*]:-}",
    "date": "$(date '+%Y-%m-%d %H:%M:%S')"
}
ENDJSON

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo " Ablation complete: $NAME"
echo "============================================================"
echo " Elapsed:       ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo " Min val BPB:   ${MIN_VAL_BPB:-n/a}"
echo " Final val BPB: ${VAL_BPB:-n/a}"
echo " CORE score:    ${CORE_SCORE:-n/a}"
echo " Num params:    ${NUM_PARAMS:-n/a}"
echo " Depth:         $ABLATION_DEPTH"
echo " Results:       $RESULT_FILE"
echo "============================================================"
