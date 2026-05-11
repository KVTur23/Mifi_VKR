#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --time=1:30:00
#SBATCH --job-name=q32b4cw_pre
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_32b_iter4_cw_pre/logs/pre_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_32b_iter4_cw_pre/logs/pre_%j.err

set -e

SRC=/mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SRC_TREE=/mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation
BASE_RUN_DIR=/mnt/pool/6/kvturanosov/VKR/finetune_runs
BASE_TMP=/tmp/kvt/qwen3_32b
MODEL_DIR=models--Qwen--Qwen3-32B
MODEL_MARKER=/tmp/kvt/models/.qwen3_32b_copy_complete

RUNS="
qwen3_32b_adalora_iter4_cw
qwen3_32b_qlora_iter4_cw
"

echo "=== qwen3_32b iter4 cw pre | $(date) ==="

mkdir -p "$BASE_RUN_DIR/qwen3_32b_iter4_cw_pre/logs"
mkdir -p /tmp/kvt/models /tmp/kvt/hf /tmp/kvt/cache /tmp/kvt/cache/tmp

has_complete_model_cache() {
    local ROOT="$1"
    local SNAPSHOTS="$ROOT/$MODEL_DIR/snapshots"

    [ -d "$SNAPSHOTS" ] || return 1
    [ -n "$(find "$SNAPSHOTS" -name config.json -exec grep -l '"model_type"' {} \; -print -quit 2>/dev/null)" ] || return 1
    [ -n "$(find "$SNAPSHOTS" -name tokenizer_config.json -print -quit 2>/dev/null)" ] || return 1
    [ -n "$(find "$SNAPSHOTS" \( -name model.safetensors -o -name model.safetensors.index.json -o -name 'model-*-of-*.safetensors' -o -name pytorch_model.bin \) -print -quit 2>/dev/null)" ] || return 1
}

if [ -f "$MODEL_MARKER" ] && has_complete_model_cache /tmp/kvt/models; then
    echo "$MODEL_DIR already copied"
else
    if [ ! -d "/mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR" ]; then
        echo "[ERROR] missing $MODEL_DIR in /mnt/pool/6/kvturanosov/VKR/models/"
        exit 1
    fi
    if ! has_complete_model_cache /mnt/pool/6/kvturanosov/VKR/models; then
        echo "[ERROR] incomplete model cache: /mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR"
        echo "        expected config.json with model_type, tokenizer_config.json and safetensors/bin weights"
        exit 1
    fi

    echo "copy/update $MODEL_DIR -> /tmp/kvt/models"
    rm -rf "/tmp/kvt/models/$MODEL_DIR"
    cp -a "/mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR" /tmp/kvt/models/
    [ -f /mnt/pool/6/kvturanosov/VKR/models/CACHEDIR.TAG ] && cp -a /mnt/pool/6/kvturanosov/VKR/models/CACHEDIR.TAG /tmp/kvt/models/
    date > "$MODEL_MARKER"
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
echo "qwen3_32b iter4 cw pre ready"
