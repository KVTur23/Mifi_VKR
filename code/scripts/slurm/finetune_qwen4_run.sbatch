#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=ft_qw4_run

set -euo pipefail

: "${BRANCH_KEY:?set BRANCH_KEY}"
: "${RUN_NAME:?set RUN_NAME}"
: "${EXPERIMENT:?set EXPERIMENT}"

GPU_PROFILE=${GPU_PROFILE:-A100_40}
TMP=/tmp/kvt/finetune_${BRANCH_KEY}_qwen4/$RUN_NAME/code

export HOME=/mnt/pool/6/kvturanosov
export XDG_CACHE_HOME=/tmp/kvt/cache
export TRITON_CACHE_DIR=/tmp/kvt/cache/triton
export TMPDIR=/tmp/kvt/cache/tmp
export HF_HOME=/tmp/kvt/hf
export HF_HUB_CACHE=/tmp/kvt/models
export HUGGINGFACE_HUB_CACHE=/tmp/kvt/models
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TMPDIR"

source /mnt/pool/6/kvturanosov/VKR/venv/bin/activate
cd "$TMP"

echo "=== run | $RUN_NAME | $EXPERIMENT | SLURM $SLURM_JOB_ID | host=$(hostname) | $(date) ==="
nvidia-smi | head -20
python -V

python -u scripts/run_finetune_experiment.py \
    --run-name "$RUN_NAME" \
    --experiment "$EXPERIMENT" \
    --gpu "$GPU_PROFILE"
