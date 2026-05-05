#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --job-name=aug_post
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/logs/aug_post_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/logs/aug_post_%j.err

set -e

echo "=== post | $(date) ==="

mkdir -p /mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/Data
mkdir -p /mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/results
mkdir -p /mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/logs

if [ -d /tmp/kvt/aug_rut5nllb_chunked_augment/code/Data ]; then
    echo "copy Data -> pool"
    cp -a /tmp/kvt/aug_rut5nllb_chunked_augment/code/Data/. \
          /mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/Data/
fi

if [ -d /tmp/kvt/aug_rut5nllb_chunked_augment/code/results ]; then
    echo "copy results -> pool"
    cp -a /tmp/kvt/aug_rut5nllb_chunked_augment/code/results/. \
          /mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/results/
fi

echo "Data:"
ls -lah /mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/Data

echo "results:"
ls -lah /mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/results
