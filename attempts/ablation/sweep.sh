#!/bin/bash
# Sweep runner: run all ablation configs (one feature at a time on top of baseline).
# Each config enables exactly ONE feature on top of the baseline d12 model.
#
# Usage:
#   bash sweep.sh                      # run all configs sequentially
#   bash sweep.sh --slurm              # submit as SLURM array job
#   bash sweep.sh --filter "item15"    # run only configs matching pattern
#   bash sweep.sh --skip-setup         # skip venv/dataset setup (faster reruns)
#   bash sweep.sh --filter "item1" --skip-setup  # combine options
#
# Output:
#   attempts/ablation/ablations/<name>/result.json   per-config results
#   attempts/ablation/ablations/sweep_summary.csv    combined CSV table

set -euo pipefail

# =============================================================================
# Resolve paths (same convention as ablation.sh / speedrun.sh)
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ABLATION_SH="$SCRIPT_DIR/ablation.sh"
ABLATIONS_DIR="$SCRIPT_DIR/ablations"

# =============================================================================
# Parse CLI arguments
# =============================================================================
SLURM_MODE=0
SKIP_SETUP=0
FILTER_PATTERN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm)
            SLURM_MODE=1
            shift
            ;;
        --skip-setup)
            SKIP_SETUP=1
            shift
            ;;
        --filter)
            FILTER_PATTERN="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash sweep.sh [--slurm] [--filter <pattern>] [--skip-setup]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Ablation defaults for d12 model
# =============================================================================
# d12 with aspect_ratio=64, head_dim=128:
#   model_dim = 12 * 64 = 768
#   n_head    = 768 / 128 = 6
#   n_kv_head (half) = 6 / 2 = 3
ABLATION_DEPTH="${ABLATION_DEPTH:-12}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
HEAD_DIM="${HEAD_DIM:-128}"
MODEL_DIM=$(( ABLATION_DEPTH * ASPECT_RATIO ))
# Round up to nearest multiple of HEAD_DIM (mirrors base_train.py logic)
MODEL_DIM=$(( ((MODEL_DIM + HEAD_DIM - 1) / HEAD_DIM) * HEAD_DIM ))
N_HEAD=$(( MODEL_DIM / HEAD_DIM ))
N_KV_HEAD_HALF=$(( N_HEAD / 2 ))

# =============================================================================
# Define all ablation configs
# Format: "name|extra_flags"
# Each config enables exactly ONE feature on top of the d12 baseline.
# =============================================================================
ALL_CONFIGS=(
    "baseline|"
    "item01_chunk_rope|--use-chunk-rope"
    "item06_fused_ce|--use-fused-ce"
    "item10a_param_rmsnorm|--use-param-rmsnorm"
    "item10b_proj_scalars|--use-proj-scalars"
    "item15_backout|--use-backout"
    "item16a_smear|--use-smear"
    "item16b_attn_gate|--use-attn-gate"
    "item18a_half_trunc_rope|--use-half-truncated-rope"
    "item18b_partial_key_offset|--use-partial-key-offset"
    "item19_bigram|--bigram-vocab-multiplier 3"
    "item20_unet_skip|--use-unet-skip"
    "item25_fused_mlp|--use-fused-mlp"
    "item30_paired_heads|--use-paired-heads"
    "item05_cautious_wd|--use-cautious-wd"
    "item07_gqa_half|--n-kv-head ${N_KV_HEAD_HALF}"
    "item09_hyperball|--use-hyperball"
    "item12_decoupled_warmdown|--matrix-warmdown-frac 1.0 --adamw-warmdown-frac 0.3"
    "item13_polar_express|--use-polar-express"
    "item14_mantissa|--use-mantissa-tracking"
    "item17_dynamic_window|--dynamic-window"
    "item21_batch_schedule|--batch-schedule 262144:0.1,524288:0.5,1048576:1.0"
    "item22_adam_every_2|--adam-every-n 2"
    "item23_mtp|--mtp-schedule 1,0.5,0.25;1,0.5;1"
    "item24_tie_embed|--tie-embed-until 0.67"
    "item27_teon|--use-teon"
    "compile_reduce_overhead|--compile-mode reduce-overhead"
)

# =============================================================================
# Apply filter if specified
# =============================================================================
CONFIGS=()
for entry in "${ALL_CONFIGS[@]}"; do
    config_name="${entry%%|*}"
    if [ -n "$FILTER_PATTERN" ]; then
        if echo "$config_name" | grep -qi "$FILTER_PATTERN"; then
            CONFIGS+=("$entry")
        fi
    else
        CONFIGS+=("$entry")
    fi
done

NUM_CONFIGS=${#CONFIGS[@]}
if [ "$NUM_CONFIGS" -eq 0 ]; then
    echo "Error: no configs match filter '$FILTER_PATTERN'"
    echo "Available configs:"
    for entry in "${ALL_CONFIGS[@]}"; do
        echo "  ${entry%%|*}"
    done
    exit 1
fi

echo "============================================================"
echo " Ablation Sweep"
echo "============================================================"
echo " Configs:     $NUM_CONFIGS"
echo " Depth:       $ABLATION_DEPTH"
echo " Model dim:   $MODEL_DIM"
echo " Num heads:   $N_HEAD"
echo " Filter:      ${FILTER_PATTERN:-<none>}"
echo " Mode:        $([ $SLURM_MODE -eq 1 ] && echo 'SLURM array job' || echo 'sequential')"
echo " Skip setup:  $([ $SKIP_SETUP -eq 1 ] && echo 'yes' || echo 'no')"
echo "============================================================"
echo ""
echo "Configs to run:"
for i in $(seq 0 $((NUM_CONFIGS - 1))); do
    entry="${CONFIGS[$i]}"
    config_name="${entry%%|*}"
    config_flags="${entry#*|}"
    printf "  [%2d] %-35s %s\n" "$i" "$config_name" "${config_flags:-<baseline>}"
done
echo ""

# =============================================================================
# SLURM mode: generate and submit an array job
# =============================================================================
if [ "$SLURM_MODE" -eq 1 ]; then
    SLURM_SCRIPT="$SCRIPT_DIR/ablations/_slurm_sweep.sh"
    CONFIGS_LIST="$SCRIPT_DIR/ablations/_slurm_configs.txt"
    mkdir -p "$SCRIPT_DIR/ablations"

    # Filter out "baseline" — it runs implicitly on every node for fair comparison
    FEATURE_CONFIGS=()
    for entry in "${CONFIGS[@]}"; do
        config_name="${entry%%|*}"
        if [ "$config_name" != "baseline" ]; then
            FEATURE_CONFIGS+=("$entry")
        fi
    done
    NUM_FEATURES=${#FEATURE_CONFIGS[@]}

    if [ "$NUM_FEATURES" -eq 0 ]; then
        echo "Error: no feature configs to submit (only baseline in list)"
        exit 1
    fi

    # Write the configs list (one per line, tab-separated name and flags)
    > "$CONFIGS_LIST"
    for entry in "${FEATURE_CONFIGS[@]}"; do
        config_name="${entry%%|*}"
        config_flags="${entry#*|}"
        printf '%s\t%s\n' "$config_name" "$config_flags" >> "$CONFIGS_LIST"
    done

    MAX_IDX=$((NUM_FEATURES - 1))

    # Partition strategy: submit to both gpu-short and gpu so SLURM picks
    # whichever has capacity first, maximizing scheduling throughput.
    # Time: 2 runs (baseline + feature) × ~5 min + overhead ≈ 15 min.
    SLURM_PARTITION="${SLURM_PARTITION:-gpu-short,gpu}"
    SLURM_TIME="${SLURM_TIME:-00:20:00}"

    cat > "$SLURM_SCRIPT" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name=ablation_sweep
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=${SLURM_TIME}
#SBATCH --array=0-${MAX_IDX}
#SBATCH --output=${SCRIPT_DIR}/ablations/slurm_%A_%a.out
#SBATCH --error=${SCRIPT_DIR}/ablations/slurm_%A_%a.err

set -euo pipefail

# Single-GPU ablations — lean on resources, just screening features
export NPROC=1

NODE_NAME=\$(hostname)
echo "========================================"
echo "Node: \$NODE_NAME"
echo "GPUs: \$(nvidia-smi -L 2>/dev/null | wc -l)x \$(nvidia-smi -L 2>/dev/null | head -1 | sed 's/GPU 0: //' | cut -d'(' -f1)"
echo "Start: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# Read config for this array task
CONFIGS_LIST="$CONFIGS_LIST"
TASK_LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "\$CONFIGS_LIST")
CONFIG_NAME=\$(echo "\$TASK_LINE" | cut -f1)
CONFIG_FLAGS=\$(echo "\$TASK_LINE" | cut -f2-)

echo ""
echo "SLURM array task \$SLURM_ARRAY_TASK_ID: feature=\$CONFIG_NAME flags=\$CONFIG_FLAGS"
echo ""

# ----------------------------------------------------------------
# Step 1: Run BASELINE on this node (for fair same-node comparison)
# ----------------------------------------------------------------
echo "========================================"
echo " Running BASELINE on \$NODE_NAME"
echo "========================================"

BASELINE_DIR="$SCRIPT_DIR/ablations/\${CONFIG_NAME}/node_baseline"
mkdir -p "\$BASELINE_DIR"

# Run baseline (output goes to feature subdir so each node has its own)
bash "$ABLATION_SH" --name "\${CONFIG_NAME}/node_baseline" \
    2>&1 | tee "\$BASELINE_DIR/log.txt"

# ----------------------------------------------------------------
# Step 2: Run FEATURE on this same node
# ----------------------------------------------------------------
echo ""
echo "========================================"
echo " Running FEATURE: \$CONFIG_NAME on \$NODE_NAME"
echo "========================================"

ABLATION_ARGS=(--name "\$CONFIG_NAME" --skip-setup)

if [ -n "\$CONFIG_FLAGS" ]; then
    ABLATION_ARGS+=(--)
    read -ra FLAG_ARRAY <<< "\$CONFIG_FLAGS"
    ABLATION_ARGS+=("\${FLAG_ARRAY[@]}")
fi

bash "$ABLATION_SH" "\${ABLATION_ARGS[@]}"

# ----------------------------------------------------------------
# Step 3: Compute same-node delta
# ----------------------------------------------------------------
BASELINE_RESULT="\$BASELINE_DIR/result.json"
FEATURE_RESULT="$SCRIPT_DIR/ablations/\${CONFIG_NAME}/result.json"
DELTA_FILE="$SCRIPT_DIR/ablations/\${CONFIG_NAME}/delta.json"

if [ -f "\$BASELINE_RESULT" ] && [ -f "\$FEATURE_RESULT" ]; then
    python3 -c "
import json, sys
with open('\$BASELINE_RESULT') as f: bl = json.load(f)
with open('\$FEATURE_RESULT') as f: ft = json.load(f)
delta = {}
delta['feature'] = ft.get('name', '')
delta['node'] = '\$NODE_NAME'
delta['baseline_min_val_bpb'] = bl.get('min_val_bpb')
delta['feature_min_val_bpb'] = ft.get('min_val_bpb')
if bl.get('min_val_bpb') is not None and ft.get('min_val_bpb') is not None:
    delta['delta_bpb'] = round(ft['min_val_bpb'] - bl['min_val_bpb'], 6)
delta['baseline_elapsed'] = bl.get('elapsed_seconds')
delta['feature_elapsed'] = ft.get('elapsed_seconds')
with open('\$DELTA_FILE', 'w') as f: json.dump(delta, f, indent=2)
print(json.dumps(delta, indent=2))
"
    echo ""
    echo "Delta saved to: \$DELTA_FILE"
fi

echo ""
echo "========================================"
echo " Job complete: \$CONFIG_NAME on \$NODE_NAME"
echo "========================================"
SLURM_EOF

    chmod +x "$SLURM_SCRIPT"
    echo "Generated SLURM array script: $SLURM_SCRIPT"
    echo "Generated configs list:        $CONFIGS_LIST"
    echo "  Feature jobs: $NUM_FEATURES (each runs baseline + feature on same node)"
    echo "  Partition:    $SLURM_PARTITION"
    echo "  Time limit:   $SLURM_TIME"
    echo ""
    echo "Submitting SLURM array job (${NUM_FEATURES} tasks)..."
    sbatch "$SLURM_SCRIPT"
    echo ""
    echo "Monitor with:  squeue -u \$USER"
    echo "Logs at:       $SCRIPT_DIR/ablations/slurm_*.out"
    echo "Results at:    $SCRIPT_DIR/ablations/<feature>/delta.json"
    echo ""
    echo "After all jobs complete, run:"
    echo "  python3 scripts/compare_ablations.py $SCRIPT_DIR/ablations/"
    exit 0
fi

# =============================================================================
# Sequential mode: run each config one at a time
# =============================================================================
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

for i in $(seq 0 $((NUM_CONFIGS - 1))); do
    entry="${CONFIGS[$i]}"
    config_name="${entry%%|*}"
    config_flags="${entry#*|}"

    echo ""
    echo "============================================================"
    echo " [$((i + 1))/$NUM_CONFIGS] Running: $config_name"
    echo "============================================================"

    # Check if result already exists (skip if re-running)
    if [ -f "$ABLATIONS_DIR/$config_name/result.json" ]; then
        echo "  Result already exists at $ABLATIONS_DIR/$config_name/result.json"
        echo "  Skipping (delete the result.json to re-run)"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    # Build ablation.sh arguments
    ABLATION_ARGS=(--name "$config_name")
    if [ "$SKIP_SETUP" -eq 1 ]; then
        ABLATION_ARGS+=(--skip-setup)
    fi

    # Only pass -- and flags if there are extra flags
    if [ -n "$config_flags" ]; then
        ABLATION_ARGS+=(--)
        # Split flags on whitespace into individual arguments
        read -ra FLAG_ARRAY <<< "$config_flags"
        ABLATION_ARGS+=("${FLAG_ARRAY[@]}")
    fi

    # Run ablation.sh and track pass/fail
    if bash "$ABLATION_SH" "${ABLATION_ARGS[@]}"; then
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  WARNING: $config_name FAILED (exit code $?)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # After the first run, skip setup for subsequent runs (venv is already active)
    SKIP_SETUP=1
done

echo ""
echo "============================================================"
echo " Sweep complete: $PASS_COUNT passed, $FAIL_COUNT failed, $SKIP_COUNT skipped"
echo "============================================================"

# =============================================================================
# Generate CSV summary from all result.json files
# =============================================================================
generate_summary() {
    local SUMMARY_CSV="$ABLATIONS_DIR/sweep_summary.csv"
    local RESULT_FILES=()

    # Collect all result.json files
    for entry in "${ALL_CONFIGS[@]}"; do
        local cname="${entry%%|*}"
        local rfile="$ABLATIONS_DIR/$cname/result.json"
        if [ -f "$rfile" ]; then
            RESULT_FILES+=("$rfile")
        fi
    done

    if [ ${#RESULT_FILES[@]} -eq 0 ]; then
        echo "No result.json files found. Skipping summary generation."
        return
    fi

    # Write CSV header
    echo "name,extra_flags,depth,nproc,min_val_bpb,final_val_bpb,core_score,num_params,elapsed_seconds,date" > "$SUMMARY_CSV"

    # Parse each result.json and append to CSV
    # Use Python for reliable JSON parsing (venv should be active from ablation runs)
    python3 -c "
import json, sys, csv, io

output = io.StringIO()
writer = csv.writer(output)

files = sys.argv[1:]
for f in files:
    try:
        with open(f) as fh:
            d = json.load(fh)
        writer.writerow([
            d.get('name', ''),
            d.get('extra_flags', ''),
            d.get('depth', ''),
            d.get('nproc', ''),
            d.get('min_val_bpb', ''),
            d.get('final_val_bpb', ''),
            d.get('core_score', ''),
            d.get('num_params', ''),
            d.get('elapsed_seconds', ''),
            d.get('date', ''),
        ])
    except Exception as e:
        print(f'Warning: failed to parse {f}: {e}', file=sys.stderr)

print(output.getvalue(), end='')
" "${RESULT_FILES[@]}" >> "$SUMMARY_CSV"

    echo ""
    echo "============================================================"
    echo " Summary CSV: $SUMMARY_CSV"
    echo "============================================================"

    # Print a formatted table to stdout
    # Get baseline bpb for delta computation
    BASELINE_BPB=""
    if [ -f "$ABLATIONS_DIR/baseline/result.json" ]; then
        BASELINE_BPB=$(python3 -c "
import json
with open('$ABLATIONS_DIR/baseline/result.json') as f:
    d = json.load(f)
print(d.get('min_val_bpb', '') or '')
")
    fi

    python3 -c "
import csv, sys

baseline_bpb = '$BASELINE_BPB'
baseline_bpb = float(baseline_bpb) if baseline_bpb else None

with open('$SUMMARY_CSV') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    print('No results to display.')
    sys.exit(0)

# Print table header
fmt = '  {:<35s} {:>12s} {:>12s} {:>10s} {:>8s} {:>10s}'
print(fmt.format('Name', 'Min Val BPB', 'Delta BPB', 'CORE', 'Time(s)', 'Params'))
print('  ' + '-' * 93)

for row in rows:
    name = row['name']
    min_bpb = row['min_val_bpb']
    core = row['core_score']
    elapsed = row['elapsed_seconds']
    params = row['num_params']

    # Compute delta from baseline
    delta = ''
    if baseline_bpb is not None and min_bpb:
        try:
            d = float(min_bpb) - baseline_bpb
            delta = f'{d:+.4f}'
        except ValueError:
            pass

    min_bpb_str = f'{float(min_bpb):.4f}' if min_bpb else 'n/a'
    core_str = f'{float(core):.4f}' if core else 'n/a'
    elapsed_str = elapsed if elapsed else 'n/a'
    params_str = params if params else 'n/a'

    print(fmt.format(name, min_bpb_str, delta if delta else 'n/a', core_str, elapsed_str, params_str))

print()
if baseline_bpb is not None:
    print(f'  Baseline min val BPB: {baseline_bpb:.4f}')
    print(f'  Negative delta = better than baseline')
print()
" 2>&1 || echo "Warning: summary table formatting failed"
}

generate_summary
