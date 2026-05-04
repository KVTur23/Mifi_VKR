#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=aug_run
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/code/logs/aug_run_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/code/logs/aug_run_%j.err

set -euo pipefail

POOL_ROOT="${POOL_ROOT:-/mnt/pool/6/kvturanosov/VKR}"
AUG_RUN_ROOT="${AUG_RUN_ROOT:-$POOL_ROOT/augmentation_runs}"
BRANCH="${BRANCH:-rut5-nllb-chunked-augment}"
RUN_NAME="${RUN_NAME:-$BRANCH}"
GPU="${GPU:-A100_40}"
CONFIG_REL="${CONFIG_REL:-config_models/aug_configs/model_vllm_32b.json}"
TMP_ROOT="${TMP_ROOT:-/tmp/kvt}"
TMP_RUN="$TMP_ROOT/augmentation_runs/$RUN_NAME"
RUN_DIR="$AUG_RUN_ROOT/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
JOB_ID="${SLURM_JOB_ID:-manual}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run_${JOB_ID}.log") 2>&1

echo "=== $(date) | augmentation run | run=$RUN_NAME | branch=$BRANCH ==="
echo "GPU=$GPU"
echo "CONFIG_REL=$CONFIG_REL"
echo "TMP_RUN=$TMP_RUN"
echo "RUN_DIR=$RUN_DIR"

export HOME=/mnt/pool/6/kvturanosov
export XDG_CACHE_HOME="$TMP_ROOT/cache"
export HF_HOME="$TMP_ROOT/hf"
export HF_HUB_CACHE="$TMP_ROOT/models"
export HUGGINGFACE_HUB_CACHE="$TMP_ROOT/models"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

source /mnt/pool/6/kvturanosov/VKR/venv/bin/activate

cd "$TMP_RUN/code"

echo "=== SLURM $JOB_ID | host=$(hostname) | $(date) ==="
nvidia-smi | head -20
python -V

python -u scripts/run_augmentation.py \
    --run-name "$RUN_NAME" \
    --gpu "$GPU" \
    --config "$CONFIG_REL" \
    ${AUG_EXTRA_ARGS:-}
