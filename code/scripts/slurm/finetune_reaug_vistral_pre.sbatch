#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --job-name=ft_vist_pre
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_vistral_24b_pre/logs/pre_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_vistral_24b_pre/logs/pre_%j.err

set -e

SRC=/mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SRC_TREE=/mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation
BASE_RUN_DIR=/mnt/pool/6/kvturanosov/VKR/finetune_runs
BASE_TMP=/tmp/kvt/finetune_reaug_vistral
MODEL_DIR=models--Vikhrmodels--Vistral-24B-Instruct
MODEL_MARKER=/tmp/kvt/models/.finetune_vistral_24b_copy_complete

RUNS="
reaug_vistral_24b_qlora_cw
reaug_vistral_24b_qlora_cw_r32
reaug_vistral_24b_qlora_cw_focal_g2
"

echo "=== common pre | Vistral-24B | $(date) ==="

mkdir -p "$BASE_RUN_DIR/reaug_vistral_24b_pre/logs"
mkdir -p /tmp/kvt/models /tmp/kvt/hf /tmp/kvt/cache /tmp/kvt/cache/tmp

if [ -f "$MODEL_MARKER" ] && \
   [ -d "/tmp/kvt/models/$MODEL_DIR" ] && \
   [ -n "$(find "/tmp/kvt/models/$MODEL_DIR/snapshots" -name config.json -print -quit 2>/dev/null)" ] && \
   [ -n "$(find "/tmp/kvt/models/$MODEL_DIR/snapshots" -name tokenizer_config.json -print -quit 2>/dev/null)" ]; then
    echo "model already copied: $MODEL_DIR"
else
    if [ ! -d "/mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR" ]; then
        echo "[ERROR] missing model cache: /mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR"
        exit 1
    fi
    if [ -z "$(find "/mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR/snapshots" -name config.json -print -quit 2>/dev/null)" ] || \
       [ -z "$(find "/mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR/snapshots" -name tokenizer_config.json -print -quit 2>/dev/null)" ]; then
        echo "[ERROR] incomplete model cache: /mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR"
        echo "        expected snapshot/config.json and snapshot/tokenizer_config.json"
        exit 1
    fi
    echo "copy/update model -> /tmp/kvt/models/$MODEL_DIR"
    mkdir -p "/tmp/kvt/models/$MODEL_DIR"
    cp -a "/mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR/." "/tmp/kvt/models/$MODEL_DIR/"
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
echo "common pre ready"
