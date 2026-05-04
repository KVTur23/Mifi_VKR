#!/bin/bash
set -euo pipefail

POOL_ROOT="${POOL_ROOT:-/mnt/pool/6/kvturanosov/VKR}"
AUG_RUN_ROOT="${AUG_RUN_ROOT:-$POOL_ROOT/augmentation_runs}"
BRANCH="${BRANCH:-rut5-nllb-chunked-augment}"
RUN_NAME="${RUN_NAME:-$BRANCH}"

if [ -z "$RUN_NAME" ]; then
    echo "RUN_NAME is empty"
    exit 1
fi

RUN_DIR="$AUG_RUN_ROOT/$RUN_NAME"
TMP_ROOT="${TMP_ROOT:-/tmp/kvt}"
TMP_RUN="$TMP_ROOT/augmentation_runs/$RUN_NAME"
TMP_CODE="$TMP_RUN/code"
LOG_DIR="$RUN_DIR/logs"
JOB_ID="${SLURM_JOB_ID:-manual}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/post_${JOB_ID}.log") 2>&1

echo "=== $(date) | augmentation post | run=$RUN_NAME ==="
echo "RUN_DIR=$RUN_DIR"
echo "TMP_CODE=$TMP_CODE"

if [ ! -d "$TMP_CODE" ]; then
    echo "[WARN] Tmp code directory not found: $TMP_CODE"
    exit 0
fi

mkdir -p "$RUN_DIR/Data" "$RUN_DIR/results" "$RUN_DIR/logs"

if [ -d "$TMP_CODE/Data" ]; then
    echo "Sync Data -> $RUN_DIR/Data"
    rsync -a "$TMP_CODE/Data/" "$RUN_DIR/Data/"
    echo "Data size: $(du -sh "$RUN_DIR/Data" | cut -f1)"
fi

if [ -d "$TMP_CODE/results" ]; then
    echo "Sync results -> $RUN_DIR/results"
    rsync -a "$TMP_CODE/results/" "$RUN_DIR/results/"
    echo "Results files: $(find "$RUN_DIR/results" -type f | wc -l)"
fi

if [ -d "$TMP_CODE/logs" ]; then
    echo "Sync code logs -> $RUN_DIR/logs/code"
    mkdir -p "$RUN_DIR/logs/code"
    rsync -a "$TMP_CODE/logs/" "$RUN_DIR/logs/code/"
fi

echo
echo "Final Data checkpoints:"
find "$RUN_DIR/Data" -maxdepth 1 -type f -name '*.csv' -printf "  %f\n" | sort || true

echo
echo "Final results:"
find "$RUN_DIR/results" -maxdepth 1 -type f -printf "  %f\n" | sort || true

if [ -f "$RUN_DIR/results/classification_results.csv" ]; then
    echo
    echo "--- classification_results.csv ---"
    cat "$RUN_DIR/results/classification_results.csv"
fi

if [ -f "$RUN_DIR/results/rubert_stage_ablation.csv" ]; then
    echo
    echo "--- rubert_stage_ablation.csv ---"
    cat "$RUN_DIR/results/rubert_stage_ablation.csv"
fi

echo "Post complete: $RUN_DIR"
