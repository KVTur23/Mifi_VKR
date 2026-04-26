#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --job-name=pre_exp3
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/code/logs/pre_exp3_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/code/logs/pre_exp3_%j.err

# Exp3-only pre-sync. Runs inside SLURM because /tmp/kvt is node-local.
# It refreshes only /tmp/kvt/exp3/code and the models needed by exp3.

set -euo pipefail

POOL=/mnt/pool/6/kvturanosov
MODELS_SRC=$POOL/VKR/models
MODELS_DST=/tmp/kvt/models
CODE_SRC=$POOL/VKR/worktrees/exp-aug-3/code
CODE_DST=/tmp/kvt/exp3/code

NEEDED_MODELS=(
  "models--Qwen--Qwen3-32B-AWQ"
  "models--google--translategemma-12b-it"
  "models--ai-forever--sbert_large_nlu_ru"
)

echo "=== SLURM $SLURM_JOB_ID | host=$(hostname) | $(date) ==="

echo "=== $(date) | PRE exp3: rsync моделей в /tmp ==="
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

echo "=== $(date) | PRE exp3: копирую worktree в /tmp ==="
if [ ! -d "$CODE_SRC" ]; then
  echo "[ERROR] worktree не найден: $CODE_SRC"
  exit 1
fi

rm -rf "$CODE_DST"
mkdir -p "$CODE_DST"
rsync -a "$CODE_SRC/" "$CODE_DST/"
mkdir -p "$CODE_DST/Data" "$CODE_DST/logs" "$CODE_DST/results" \
  /tmp/kvt/exp3/home /tmp/kvt/exp3/hf /tmp/kvt/exp3/cache /tmp/kvt/exp3/triton

rm -f "$CODE_DST/Data/_stage3_pairs_cache.csv" "$CODE_DST/Data/data_after_stage3.csv"

echo "=== code check ==="
grep -n "translategemma\\|source_lang_code\\|target_lang_code" \
  "$CODE_DST/src/augmentation/stage3_back_translation.py" | head -20

echo "=== data check ==="
ls -lh "$CODE_DST/Data/data_after_stage1.csv" "$CODE_DST/Data/data_after_stage2.csv"

echo "=== $(date) | PRE exp3: DONE ==="
