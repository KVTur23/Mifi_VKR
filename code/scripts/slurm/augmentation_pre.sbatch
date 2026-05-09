#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --time=1:30:00
#SBATCH --job-name=aug_pre
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/augmentation_runs/aug_full/logs/pre_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/augmentation_runs/aug_full/logs/pre_%j.err

set -e

SRC=/mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SRC_TREE=/mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation
RUN_NAME=aug_full
RUN_DIR=/mnt/pool/6/kvturanosov/VKR/augmentation_runs/$RUN_NAME
TMP=/tmp/kvt/aug_full/code

# Список моделей под аугментацию. Структура HF cache: models--<org>--<name>/
# - Qwen2.5-32B-AWQ - vLLM stage1 (LLM-генерация) и stage2 (парафраз через тот же vLLM)
# - NLLB-200-3.3B   - stage3 back-translation
# - SBERT           - валидация (cosine similarity)
# - rubert-tiny2/base - baseline и augmented metrics
MODELS="
models--Qwen--Qwen2.5-32B-Instruct-AWQ
models--facebook--nllb-200-3.3B
models--ai-forever--sbert_large_nlu_ru
models--cointegrated--rubert-tiny2
models--DeepPavlov--rubert-base-cased
"

echo "=== aug pre | $RUN_NAME | $(date) ==="

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/results" "$RUN_DIR/Data"
mkdir -p /tmp/kvt/models /tmp/kvt/hf /tmp/kvt/cache

# Копируем модели из persistent в /tmp scratch (один раз на ноду — маркером)
MARKER=/tmp/kvt/models/.aug_full_copy_complete
if [ ! -f "$MARKER" ]; then
    echo "copy models -> /tmp/kvt/models"
    for MODEL_DIR in $MODELS; do
        if [ -d "/mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR" ]; then
            echo "copy $MODEL_DIR"
            cp -a "/mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR" /tmp/kvt/models/
        else
            echo "[WARN] missing $MODEL_DIR (запуск скачает онлайн если есть интернет)"
        fi
    done
    [ -f /mnt/pool/6/kvturanosov/VKR/models/CACHEDIR.TAG ] && cp -a /mnt/pool/6/kvturanosov/VKR/models/CACHEDIR.TAG /tmp/kvt/models/
    date > "$MARKER"
else
    echo "augmentation models already copied"
fi

echo "copy code -> $TMP"
rm -rf /tmp/kvt/aug_full
mkdir -p "$TMP"
cp -a "$SRC/." "$TMP/"

rm -rf "$TMP/Data" "$TMP/results" "$TMP/logs"
mkdir -p "$TMP/Data" "$TMP/results" "$TMP/logs"

# Если RUN_DIR/Data пуст — копируем из ветки (там data.json + EDA при первом запуске)
if [ -z "$(find "$RUN_DIR/Data" -mindepth 1 -maxdepth 1 2>/dev/null)" ]; then
    echo "copy branch Data -> run"
    cp -a "$SRC/Data/." "$RUN_DIR/Data/"
fi

echo "copy run Data -> tmp"
cp -a "$RUN_DIR/Data/." "$TMP/Data/"

git -C "$SRC_TREE" rev-parse HEAD > "$TMP/source_commit.txt" 2>/dev/null || true
ls -lah "$TMP/Data"

echo
echo "aug pre ready"
