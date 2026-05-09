#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=q32_ada_cw
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_32b_adalora_cw/logs/run_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_32b_adalora_cw/logs/run_%j.err

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
cd /tmp/kvt/qwen3_32b/qwen3_32b_adalora_cw/code

echo "=== run | qwen3_32b_adalora_cw | SLURM $SLURM_JOB_ID | host=$(hostname) | $(date) ==="
nvidia-smi | head -20
python -V

python -u scripts/run_finetune.py \
    --config qwen3_32b_adalora_cw.json \
    --gpu A100_40
