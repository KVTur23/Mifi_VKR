#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=aug_metrics
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/code/logs/aug_metrics_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/code/logs/aug_metrics_%j.err

# Метрики до/после аугментации для одного exp-worktree.
# Запуск:
#   sbatch --export=EXP_NUM=4 scripts/slurm/aug_metrics.sbatch
#
# Ожидает, что aug_pre.sbatch уже скопировал worktree в /tmp/kvt/exp${EXP_NUM}/code.

set -e

POOL=/mnt/pool/6/kvturanosov
EXP_NUM=${EXP_NUM:-4}
CODE_DST=${CODE_DST:-/tmp/kvt/exp${EXP_NUM}/code}
EXP_NAME=${EXP_NAME:-exp${EXP_NUM}}

export PYTHONUNBUFFERED=1
export HOME=/tmp/kvt/exp${EXP_NUM}/home
export XDG_CACHE_HOME=/tmp/kvt/exp${EXP_NUM}/cache
export HF_HOME=/tmp/kvt/exp${EXP_NUM}/hf

mkdir -p "$HOME" "$XDG_CACHE_HOME" "$HF_HOME"

echo "=== SLURM $SLURM_JOB_ID | host=$(hostname) | $(date) ==="
nvidia-smi | head -20 || true

source $POOL/VKR/venv/bin/activate
cd $CODE_DST

echo "=== $(date) | AUG METRICS: $EXP_NAME ==="
python -u scripts/run_augmentation_metrics.py \
  --experiment "$EXP_NAME" \
  --results-dir results \
  --after-stage 3 \
  --force

echo "=== $(date) | POST: rsync metrics в pool ==="
RESULTS_DST=$POOL/VKR/results/aug_exp${EXP_NUM}
mkdir -p $RESULTS_DST
rsync -a results/ $RESULTS_DST/
echo "results: $(ls $RESULTS_DST | wc -l) файлов"

echo "=== $(date) | aug_metrics DONE ==="
