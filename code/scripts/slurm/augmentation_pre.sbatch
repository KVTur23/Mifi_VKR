#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --job-name=aug_pre
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/logs/aug_pre_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/logs/aug_pre_%j.err

set -e

echo "=== pre | $(date) ==="

mkdir -p /mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/Data
mkdir -p /mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/results
mkdir -p /tmp/kvt/models /tmp/kvt/hf /tmp/kvt/cache
mkdir -p /tmp/kvt/cache/tmp /tmp/kvt/cache/triton /tmp/kvt/cache/torchinductor /tmp/kvt/cache/cuda

if [ ! -f /tmp/kvt/models/.augmentation_copy_complete ]; then
    echo "copy needed augmentation models -> /tmp/kvt/models"
    for MODEL_DIR in \
        models--Qwen--Qwen2.5-32B-Instruct-AWQ \
        models--fyaronskiy--ruT5-large-paraphraser \
        models--facebook--nllb-200-3.3B \
        models--facebook--nllb-200-distilled-1.3B \
        models--ai-forever--sbert_large_nlu_ru
    do
        if [ -d "/mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR" ]; then
            echo "copy $MODEL_DIR"
            cp -a "/mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR" /tmp/kvt/models/
        else
            echo "missing $MODEL_DIR"
        fi
    done
    [ -f /mnt/pool/6/kvturanosov/VKR/models/CACHEDIR.TAG ] && cp -a /mnt/pool/6/kvturanosov/VKR/models/CACHEDIR.TAG /tmp/kvt/models/
    date > /tmp/kvt/models/.augmentation_copy_complete
else
    echo "augmentation models already copied"
fi

echo "prepare /tmp/kvt/aug_rut5nllb_chunked_augment/code"
rm -rf /tmp/kvt/aug_rut5nllb_chunked_augment
mkdir -p /tmp/kvt/aug_rut5nllb_chunked_augment/code

cp -a /mnt/pool/6/kvturanosov/VKR/worktrees/rut5-nllb-chunked-augment/code/. \
      /tmp/kvt/aug_rut5nllb_chunked_augment/code/

rm -rf /tmp/kvt/aug_rut5nllb_chunked_augment/code/Data
rm -rf /tmp/kvt/aug_rut5nllb_chunked_augment/code/results
rm -rf /tmp/kvt/aug_rut5nllb_chunked_augment/code/logs
mkdir -p /tmp/kvt/aug_rut5nllb_chunked_augment/code/Data
mkdir -p /tmp/kvt/aug_rut5nllb_chunked_augment/code/results
mkdir -p /tmp/kvt/aug_rut5nllb_chunked_augment/code/logs

if [ -n "$(find /mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/Data -mindepth 1 -maxdepth 1 2>/dev/null)" ]; then
    echo "copy existing run Data -> tmp"
    cp -a /mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/Data/. \
          /tmp/kvt/aug_rut5nllb_chunked_augment/code/Data/
else
    echo "copy branch Data -> run and tmp"
    cp -a /mnt/pool/6/kvturanosov/VKR/worktrees/rut5-nllb-chunked-augment/code/Data/. \
          /mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/Data/
    cp -a /mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/Data/. \
          /tmp/kvt/aug_rut5nllb_chunked_augment/code/Data/
fi

echo "ready"
ls -lah /tmp/kvt/aug_rut5nllb_chunked_augment/code/Data
