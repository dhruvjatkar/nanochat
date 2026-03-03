#!/bin/bash
# Run attempt 002 followed by baseline 001 on the same node.
# Ensures fair comparison by using the same physical hardware.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "========================================"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L | wc -l)x $(nvidia-smi -L | head -1 | sed 's/GPU [0-9]*: //' | sed 's/ (.*//')"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

echo ""
echo "========================================"
echo "  ATTEMPT 002 — Starting"
echo "========================================"
START_002=$(date +%s)
bash "$REPO_ROOT/attempts/002/speedrun.sh"
END_002=$(date +%s)
echo "========================================"
echo "  ATTEMPT 002 — Finished in $(( (END_002 - START_002) / 60 ))m $(( (END_002 - START_002) % 60 ))s"
echo "========================================"

# Save attempt 002 report before baseline overwrites shared state
cp "$REPO_ROOT/attempts/002/report.md" "$REPO_ROOT/attempts/002/report_002.md" 2>/dev/null || true

echo ""
echo "========================================"
echo "  BASELINE 001 — Starting"
echo "========================================"
START_001=$(date +%s)
bash "$REPO_ROOT/attempts/001-baseline/speedrun.sh"
END_001=$(date +%s)
echo "========================================"
echo "  BASELINE 001 — Finished in $(( (END_001 - START_001) / 60 ))m $(( (END_001 - START_001) % 60 ))s"
echo "========================================"

echo ""
echo "========================================"
echo "  SUMMARY"
echo "  Attempt 002: $(( (END_002 - START_002) / 60 ))m $(( (END_002 - START_002) % 60 ))s"
echo "  Baseline 001: $(( (END_001 - START_001) / 60 ))m $(( (END_001 - START_001) % 60 ))s"
echo "  Node: $(hostname)"
echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
