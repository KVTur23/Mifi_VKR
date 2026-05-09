#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --job-name=aug_run
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/augmentation_runs/aug_full/logs/run_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/augmentation_runs/aug_full/logs/run_%j.err

set -e

mkdir -p /tmp/kvt/cache/tmp /tmp/kvt/cache/triton /tmp/kvt/cache/torchinductor /tmp/kvt/cache/cuda

export HOME=/mnt/pool/6/kvturanosov
export TMPDIR=/tmp/kvt/cache/tmp
export XDG_CACHE_HOME=/tmp/kvt/cache
export HF_HOME=/tmp/kvt/hf
export HF_HUB_CACHE=/tmp/kvt/models
export HUGGINGFACE_HUB_CACHE=/tmp/kvt/models
export TRITON_CACHE_DIR=/tmp/kvt/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/kvt/cache/torchinductor
export CUDA_CACHE_PATH=/tmp/kvt/cache/cuda
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /mnt/pool/6/kvturanosov/VKR/venv/bin/activate
cd /tmp/kvt/aug_full/code

echo "=== aug run | aug_full | SLURM $SLURM_JOB_ID | host=$(hostname) | $(date) ==="
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
echo "CUDA_CACHE_PATH=$CUDA_CACHE_PATH"
nvidia-smi | head -20
python -V

python -u scripts/augmentation.py \
    --gpu A100_40 \
    --config config_models/aug_configs/model_vllm_32b.json
