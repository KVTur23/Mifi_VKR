#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --job-name=aug_pre
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/code/logs/aug_pre_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/code/logs/aug_pre_%j.err

# aug_pre.sbatch — предварительная синхронизация для всех 4 aug-экспериментов.
# 1) rsync всех нужных моделей из pool в /tmp/kvt/models (shared read-only для 4 джобов)
# 2) cp 4 worktree-копии кода в /tmp/kvt/exp{1..4}/code
# Comp-джобы (aug_exp{1..4}.sbatch) после этого стартуют параллельно через --dependency.
# Лежит только в ветке exp-aug-1 — одну общую версию удобнее поддерживать.

set -e
POOL=/mnt/pool/6/kvturanosov
MODELS_SRC=$POOL/VKR/models
MODELS_DST=/tmp/kvt/models

# все модели, нужные хотя бы одному эксперименту
NEEDED_MODELS=(
  "models--Qwen--Qwen3-32B-AWQ"
  "models--Qwen--Qwen2.5-14B-Instruct-AWQ"
  "models--qdzzzxc--RuadaptQwen3-32B-Instruct-AWQ"
  "models--google--translategemma-12b-it"
  "models--mradermacher--YanoljaNEXT-Rosetta-20B-GGUF"
  "models--fyaronskiy--ruT5-large-paraphraser"
  "models--facebook--nllb-200-3.3B"
  "models--ai-forever--sbert_large_nlu_ru"
)

echo "=== $(date) | PRE: rsync моделей в /tmp ==="
mkdir -p $MODELS_DST
for m in "${NEEDED_MODELS[@]}"; do
  if [ -d "$MODELS_SRC/$m" ]; then
    echo "  rsync $m"
    rsync -a "$MODELS_SRC/$m/" "$MODELS_DST/$m/"
  else
    echo "  [WARN] $m не найдена в pool — пропускаю"
  fi
done
echo "Модели в /tmp: $(du -sh $MODELS_DST | cut -f1)"

echo "=== $(date) | PRE: копирую 4 worktree в /tmp ==="
for N in 1 2 3 4; do
  SRC=$POOL/VKR/worktrees/exp-aug-$N/code
  DST=/tmp/kvt/exp$N/code
  if [ ! -d "$SRC" ]; then
    echo "  [WARN] worktree $SRC не найден — пропускаю exp$N"
    continue
  fi
  rm -rf "$DST"
  mkdir -p "$DST"
  cp -r "$SRC/." "$DST/"
  mkdir -p "$DST/Data" "$DST/logs" "$DST/results" "/tmp/kvt/exp$N/hf" "/tmp/kvt/exp$N/cache"
  echo "  exp$N: $(du -sh $DST | cut -f1)"
done

echo "=== $(date) | PRE: DONE ==="
