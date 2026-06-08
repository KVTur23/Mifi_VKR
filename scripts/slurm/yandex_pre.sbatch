#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --job-name=yandex_pre
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/finetune_runs/yandex_pre/logs/pre_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/finetune_runs/yandex_pre/logs/pre_%j.err

set -e

SRC=/mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SRC_TREE=/mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation
BASE_RUN_DIR=/mnt/pool/6/kvturanosov/VKR/finetune_runs
BASE_TMP=/tmp/kvt/yandex
MODEL_DIR=models--yandex--YandexGPT-5-Lite-8B-instruct
MODEL_MARKER=/tmp/kvt/models/.yandex_5_lite_8b_copy_complete

# 4 PEFT-метода для Yandex-8B (iter1 = базовые гиперы r=16/alpha=32, без CW).
# Все 4 запускаются параллельно после общего pre.
RUNS="
yandexgpt_5_lite_8b_lora_iter1
yandexgpt_5_lite_8b_qlora_iter1
yandexgpt_5_lite_8b_adalora_iter1
yandexgpt_5_lite_8b_tinylora_iter1
"

echo "=== yandex pre | $(date) ==="

mkdir -p "$BASE_RUN_DIR/yandex_pre/logs"
mkdir -p /tmp/kvt/models /tmp/kvt/hf /tmp/kvt/cache /tmp/kvt/cache/tmp

if [ ! -f "$MODEL_MARKER" ]; then
    if [ -d "/mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR" ]; then
        echo "copy $MODEL_DIR -> /tmp/kvt/models"
        cp -a "/mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR" /tmp/kvt/models/
        [ -f /mnt/pool/6/kvturanosov/VKR/models/CACHEDIR.TAG ] && cp -a /mnt/pool/6/kvturanosov/VKR/models/CACHEDIR.TAG /tmp/kvt/models/
        date > "$MODEL_MARKER"
    else
        echo "[ERROR] missing $MODEL_DIR в /mnt/pool/6/kvturanosov/VKR/models/"
        echo "        Скачай через: source venv && HF_HUB_CACHE=/mnt/pool/.../models huggingface-cli download yandex/YandexGPT-5-Lite-8B-instruct"
        exit 1
    fi
else
    echo "yandex 8B model already copied"
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
echo "yandex pre ready"
