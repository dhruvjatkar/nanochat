#!/bin/bash
#SBATCH --job-name=ablation_sweep
#SBATCH --partition=gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --array=0-6
#SBATCH --output=/projects/Sontag_Lab_Storage/nanochat/attempts/ablation/ablations/slurm_%A_%a.out
#SBATCH --error=/projects/Sontag_Lab_Storage/nanochat/attempts/ablation/ablations/slurm_%A_%a.err

set -euo pipefail

# Single-GPU ablations — lean on resources, just screening features
export NPROC=1
# Reduced data ratio for fast screening (~420 steps instead of 2205)
export TARGET_PARAM_DATA_RATIO=2.0

NODE_NAME=$(hostname)
CHUNK_FILE="/projects/Sontag_Lab_Storage/nanochat/attempts/ablation/ablations/_slurm_chunks/chunk_${SLURM_ARRAY_TASK_ID}.txt"

echo "========================================"
echo "Node: $NODE_NAME"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 | sed 's/GPU 0: //' | cut -d'(' -f1)"
echo "Chunk: $SLURM_ARRAY_TASK_ID"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""
echo "Features in this chunk:"
cat "$CHUNK_FILE"
echo ""

# ----------------------------------------------------------------
# Step 1: Run BASELINE on this node (shared by all features in chunk)
# ----------------------------------------------------------------
echo "========================================"
echo " Running BASELINE on $NODE_NAME"
echo "========================================"

BASELINE_NAME="chunk_${SLURM_ARRAY_TASK_ID}_baseline"
bash "/projects/Sontag_Lab_Storage/nanochat/attempts/ablation/ablation.sh" --name "$BASELINE_NAME"

BASELINE_RESULT="/projects/Sontag_Lab_Storage/nanochat/attempts/ablation/ablations/${BASELINE_NAME}/result.json"

# ----------------------------------------------------------------
# Step 2: Run each FEATURE in this chunk, then compute delta
# ----------------------------------------------------------------
while IFS=$'\t' read -r CONFIG_NAME CONFIG_FLAGS; do
    echo ""
    echo "========================================"
    echo " Running FEATURE: $CONFIG_NAME on $NODE_NAME"
    echo "========================================"

    ABLATION_ARGS=(--name "$CONFIG_NAME" --skip-setup)

    if [ -n "$CONFIG_FLAGS" ]; then
        ABLATION_ARGS+=(--)
        read -ra FLAG_ARRAY <<< "$CONFIG_FLAGS"
        ABLATION_ARGS+=("${FLAG_ARRAY[@]}")
    fi

    bash "/projects/Sontag_Lab_Storage/nanochat/attempts/ablation/ablation.sh" "${ABLATION_ARGS[@]}" || {
        echo "WARNING: $CONFIG_NAME FAILED"
        continue
    }

    # Compute same-node delta
    FEATURE_RESULT="/projects/Sontag_Lab_Storage/nanochat/attempts/ablation/ablations/${CONFIG_NAME}/result.json"
    DELTA_FILE="/projects/Sontag_Lab_Storage/nanochat/attempts/ablation/ablations/${CONFIG_NAME}/delta.json"

    if [ -f "$BASELINE_RESULT" ] && [ -f "$FEATURE_RESULT" ]; then
        python3 -c "
import json
with open('$BASELINE_RESULT') as f: bl = json.load(f)
with open('$FEATURE_RESULT') as f: ft = json.load(f)
delta = {
    'feature': ft.get('name', ''),
    'node': '$NODE_NAME',
    'baseline_min_val_bpb': bl.get('min_val_bpb'),
    'feature_min_val_bpb': ft.get('min_val_bpb'),
    'baseline_elapsed': bl.get('elapsed_seconds'),
    'feature_elapsed': ft.get('elapsed_seconds'),
}
if bl.get('min_val_bpb') is not None and ft.get('min_val_bpb') is not None:
    delta['delta_bpb'] = round(ft['min_val_bpb'] - bl['min_val_bpb'], 6)
with open('$DELTA_FILE', 'w') as f: json.dump(delta, f, indent=2)
print(json.dumps(delta, indent=2))
"
    fi

done < "$CHUNK_FILE"

echo ""
echo "========================================"
echo " Chunk $SLURM_ARRAY_TASK_ID complete on $NODE_NAME"
echo "========================================"
