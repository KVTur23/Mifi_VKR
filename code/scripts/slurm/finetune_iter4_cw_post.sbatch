#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --time=1:00:00

set -e

if [ -z "$RUN_NAME" ]; then
    echo "[ERROR] RUN_NAME must be set"
    exit 1
fi

BASE_RUN_DIR=/mnt/pool/6/kvturanosov/VKR/finetune_runs
BASE_TMP=/tmp/kvt/${TMP_GROUP:-iter4_cw}
RUN_DIR="$BASE_RUN_DIR/$RUN_NAME"
TMP="$BASE_TMP/$RUN_NAME/code"

echo "=== iter4 cw post | $RUN_NAME | $(date) ==="

mkdir -p "$RUN_DIR/results" "$RUN_DIR/logs" "$RUN_DIR/adapter"

if [ -d "$TMP/results" ]; then
    cp -a "$TMP/results/." "$RUN_DIR/results/"
fi

if [ -d "$TMP/Data/finetune_checkpoints/$RUN_NAME" ]; then
    rm -rf "$RUN_DIR/adapter"
    mkdir -p "$RUN_DIR/adapter"
    cp -a "$TMP/Data/finetune_checkpoints/$RUN_NAME/." "$RUN_DIR/adapter/"
fi

echo "results:"
ls -lah "$RUN_DIR/results"
echo "adapter:"
ls -lah "$RUN_DIR/adapter"
