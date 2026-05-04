#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=ft_run_r32
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_qwen3_32b_qlora_cw_r32/logs/run_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_qwen3_32b_qlora_cw_r32/logs/run_%j.err

set -e

export HOME=/mnt/pool/6/kvturanosov
export XDG_CACHE_HOME=/tmp/kvt/cache
export HF_HOME=/tmp/kvt/hf
export HF_HUB_CACHE=/tmp/kvt/models
export HUGGINGFACE_HUB_CACHE=/tmp/kvt/models
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /mnt/pool/6/kvturanosov/VKR/venv/bin/activate
cd /tmp/kvt/finetune_reaug/reaug_qwen3_32b_qlora_cw_r32/code

echo "=== run | reaug_qwen3_32b_qlora_cw_r32 | SLURM $SLURM_JOB_ID | host=$(hostname) | $(date) ==="
nvidia-smi | head -20
python -V

python -u scripts/run_finetune_experiment.py \
    --run-name reaug_qwen3_32b_qlora_cw_r32 \
    --experiment cw_r32 \
    --gpu A100_40
