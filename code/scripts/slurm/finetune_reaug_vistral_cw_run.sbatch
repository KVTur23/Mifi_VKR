#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=ft_vist_cw
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_vistral_24b_qlora_cw/logs/run_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_vistral_24b_qlora_cw/logs/run_%j.err

set -e

mkdir -p /tmp/kvt/cache/tmp /tmp/kvt/cache/triton /tmp/kvt/cache/torchinductor /tmp/kvt/cache/cuda

export HOME=/mnt/pool/6/kvturanosov
export TMPDIR=/tmp/kvt/cache/tmp
export XDG_CACHE_HOME=/tmp/kvt/cache
export HF_HOME=/tmp/kvt/hf
export HF_HUB_CACHE=/tmp/kvt/models
export HUGGINGFACE_HUB_CACHE=/tmp/kvt/models
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/kvt/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/kvt/cache/torchinductor
export CUDA_CACHE_PATH=/tmp/kvt/cache/cuda
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /mnt/pool/6/kvturanosov/VKR/venv/bin/activate
cd /tmp/kvt/finetune_reaug_vistral/reaug_vistral_24b_qlora_cw/code

echo "=== run | reaug_vistral_24b_qlora_cw | SLURM $SLURM_JOB_ID | host=$(hostname) | $(date) ==="
nvidia-smi | head -20
python -V

python -u scripts/run_finetune_experiment.py \
    --run-name reaug_vistral_24b_qlora_cw \
    --experiment cw \
    --base-config config_models/finetune_configs/vistral_24b_qlora.json \
    --gpu A100_40
