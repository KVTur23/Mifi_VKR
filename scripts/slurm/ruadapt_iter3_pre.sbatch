#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --time=1:30:00
#SBATCH --job-name=ruad3_pre
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/finetune_runs/ruadapt_iter3_pre/logs/pre_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/finetune_runs/ruadapt_iter3_pre/logs/pre_%j.err

set -e

SRC=/mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SRC_TREE=/mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation
BASE_RUN_DIR=/mnt/pool/6/kvturanosov/VKR/finetune_runs
BASE_TMP=/tmp/kvt/ruadapt
MODEL_DIR=models--RefalMachine--RuadaptQwen3-32B-Instruct
MODEL_MARKER=/tmp/kvt/models/.ruadapt_qwen3_32b_copy_complete

RUNS="
ruadapt_qwen3_32b_qlora_iter3
ruadapt_qwen3_32b_adalora_iter3
"

echo "=== ruadapt_iter3 pre | $(date) ==="

mkdir -p "$BASE_RUN_DIR/ruadapt_iter3_pre/logs"
mkdir -p /tmp/kvt/models /tmp/kvt/hf /tmp/kvt/cache /tmp/kvt/cache/tmp

if [ ! -f "$MODEL_MARKER" ]; then
    if [ -d "/mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR" ]; then
        echo "copy $MODEL_DIR -> /tmp/kvt/models"
        cp -a "/mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR" /tmp/kvt/models/
        [ -f /mnt/pool/6/kvturanosov/VKR/models/CACHEDIR.TAG ] && cp -a /mnt/pool/6/kvturanosov/VKR/models/CACHEDIR.TAG /tmp/kvt/models/
        date > "$MODEL_MARKER"
    else
        echo "[ERROR] missing $MODEL_DIR в /mnt/pool/6/kvturanosov/VKR/models/"
        exit 1
    fi
else
    echo "ruadapt-32B model already copied"
fi

for RUN_NAME in $RUNS; do
    RUN_DIR="$BASE_RUN_DIR/$RUN_NAME"
    TMP="$BASE_TMP/$RUN_NAME/code"

    echo
    echo "=== prepare $RUN_NAME ==="
    mkdir -p "$RUN_DIR/logs" "$RUN_DIR/results" "$RUN_DIR/Data"

    echo "copy code -> $TMP"
    rm -rf "$BASE_TMP/$RUN_NAME"
    mkdir -p "$TMP"
    cp -a "$SRC/." "$TMP/"

    rm -rf "$TMP/Data" "$TMP/results" "$TMP/logs"
    mkdir -p "$TMP/Data" "$TMP/results" "$TMP/logs" "$TMP/Data/finetune_checkpoints"

    if [ -z "$(find "$RUN_DIR/Data" -mindepth 1 -maxdepth 1 2>/dev/null)" ]; then
        echo "copy branch Data -> run"
        cp -a "$SRC/Data/." "$RUN_DIR/Data/"
    fi

    echo "copy run Data -> tmp"
    cp -a "$RUN_DIR/Data/." "$TMP/Data/"

    git -C "$SRC_TREE" rev-parse HEAD > "$TMP/source_commit.txt" 2>/dev/null || true
    ls -lah "$TMP/Data"
done

echo
echo "ruadapt_iter3 pre ready"
