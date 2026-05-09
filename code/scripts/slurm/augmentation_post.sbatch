#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --job-name=aug_post
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/augmentation_runs/aug_full/logs/post_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/augmentation_runs/aug_full/logs/post_%j.err

set -e

RUN_NAME=aug_full
RUN_DIR=/mnt/pool/6/kvturanosov/VKR/augmentation_runs/$RUN_NAME
TMP=/tmp/kvt/aug_full/code

echo "=== aug post | $RUN_NAME | $(date) ==="

mkdir -p "$RUN_DIR/Data" "$RUN_DIR/results" "$RUN_DIR/logs"

# Аугментированные данные (data_after_stage1/2/3.csv + кэши пар) - на pool
if [ -d "$TMP/Data" ]; then
    cp -a "$TMP/Data/." "$RUN_DIR/Data/"
fi

# Метрики baseline + augmented (classification_results.csv)
if [ -d "$TMP/results" ]; then
    cp -a "$TMP/results/." "$RUN_DIR/results/"
fi

echo "Data:"
ls -lah "$RUN_DIR/Data"
echo "results:"
ls -lah "$RUN_DIR/results"
