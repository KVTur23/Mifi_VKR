#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --job-name=aug_pre_nllb3
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/code/logs/aug_pre_nllb3_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/code/logs/aug_pre_nllb3_%j.err

# aug_pre_nllb3.sbatch — быстрая pre-синхронизация только для NLLB-экспериментов 1, 2 и 4.
# Копирует нужные модели и worktree-код в /tmp/kvt/exp{1,2,4}/code.
# Stage3-кэш в /tmp очищается, чтобы обратный перевод стартовал заново на свежем коде.

set -euo pipefail

POOL=/mnt/pool/6/kvturanosov
MODELS_SRC=$POOL/VKR/models
MODELS_DST=/tmp/kvt/models

NEEDED_MODELS=(
  "models--Qwen--Qwen3-32B-AWQ"
  "models--qdzzzxc--RuadaptQwen3-32B-Instruct-AWQ"
  "models--fyaronskiy--ruT5-large-paraphraser"
  "models--facebook--nllb-200-distilled-1.3B"
  "models--ai-forever--sbert_large_nlu_ru"
)

echo "=== $(date) | PRE NLLB3: rsync моделей в /tmp ==="
mkdir -p "$MODELS_DST"
for m in "${NEEDED_MODELS[@]}"; do
  if [ -d "$MODELS_SRC/$m" ]; then
    echo "  rsync $m"
    rsync -a "$MODELS_SRC/$m/" "$MODELS_DST/$m/"
  else
    echo "  [WARN] $m не найдена в pool — пропускаю"
  fi
done
echo "Модели в /tmp: $(du -sh "$MODELS_DST" | cut -f1)"

echo "=== $(date) | PRE NLLB3: копирую worktree exp1/2/4 в /tmp ==="
for N in 1 2 4; do
  SRC=$POOL/VKR/worktrees/exp-aug-$N/code
  DST=/tmp/kvt/exp$N/code
  if [ ! -d "$SRC" ]; then
    echo "  [WARN] worktree $SRC не найден — пропускаю exp$N"
    continue
  fi

  rm -rf "$DST"
  mkdir -p "$DST"
  rsync -a "$SRC/" "$DST/"
  mkdir -p "$DST/Data" "$DST/logs" "$DST/results" "/tmp/kvt/exp$N/hf" "/tmp/kvt/exp$N/cache"

  rm -f "$DST/Data/data_after_stage3.csv" "$DST/Data/_stage3_pairs_cache.csv"

  echo "  exp$N: $(du -sh "$DST" | cut -f1)"
  grep -n "MODEL_NLLB\\|MAX_NLLB_BATCH_SIZE\\|NLLB_CHUNK_CHARS" \
    "$DST/src/augmentation/stage3_back_translation.py" || true
  grep -n "nllb_model\\|nllb_batch_size\\|nllb_chunk_chars" \
    "$DST/config_models/pipeline_config.json" || true
done

echo "=== $(date) | PRE NLLB3: DONE ==="
