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
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/logs/aug_run_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/logs/aug_run_%j.err

set -e

export HOME=/mnt/pool/6/kvturanosov
export XDG_CACHE_HOME=/tmp/kvt/cache
export HF_HOME=/tmp/kvt/hf
export HF_HUB_CACHE=/tmp/kvt/models
export HUGGINGFACE_HUB_CACHE=/tmp/kvt/models
export AUG_POOL_DATA_DIR=/mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/Data
export TMPDIR=/tmp/kvt/cache/tmp
export TRITON_CACHE_DIR=/tmp/kvt/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/kvt/cache/torchinductor
export CUDA_CACHE_PATH=/tmp/kvt/cache/cuda
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$AUG_POOL_DATA_DIR" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"

source /mnt/pool/6/kvturanosov/VKR/venv/bin/activate
cd /tmp/kvt/aug_rut5nllb_chunked_augment/code

echo "=== run | SLURM $SLURM_JOB_ID | host=$(hostname) | $(date) ==="
nvidia-smi | head -20
python -V

python -u scripts/run_augmentation.py \
    --run-name rut5nllb_chunked_augment \
    --gpu A100_40 \
    --config config_models/aug_configs/model_vllm_32b.json
