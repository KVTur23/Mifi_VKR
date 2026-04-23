#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=ft_adalora
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/code/logs/ft_adalora_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/code/logs/ft_adalora_%j.err

set -e

export HOME=/mnt/pool/6/kvturanosov
export XDG_CACHE_HOME=/tmp/kvt/cache
export HF_HOME=/tmp/kvt/hf
export HF_HUB_CACHE=/tmp/kvt/models
export HUGGINGFACE_HUB_CACHE=/tmp/kvt/models
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

source /mnt/pool/6/kvturanosov/VKR/venv/bin/activate
cd /tmp/kvt/code

echo "=== SLURM $SLURM_JOB_ID | host=$(hostname) | $(date) ==="
nvidia-smi | head -20

python -u scripts/run_finetune.py --methods adalora_qwen3_32b --gpu A100_40
