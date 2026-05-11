#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --job-name=q14_a3_post
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_14b_adalora_iter3/logs/post_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_14b_adalora_iter3/logs/post_%j.err

set -e

RUN_NAME=qwen3_14b_adalora_iter3
RUN_DIR=/mnt/pool/6/kvturanosov/VKR/finetune_runs/$RUN_NAME
TMP=/tmp/kvt/qwen3_14b/$RUN_NAME/code

echo "=== post | $RUN_NAME | $(date) ==="

mkdir -p "$RUN_DIR/results" "$RUN_DIR/logs" "$RUN_DIR/adapter"

if [ -d "$TMP/results" ]; then
    cp -a "$TMP/results/." "$RUN_DIR/results/"
fi

if [ -d "$TMP/Data/finetune_checkpoints/$RUN_NAME" ]; then
    rm -rf "$RUN_DIR/adapter"
    mkdir -p "$RUN_DIR/adapter"
    cp -a "$TMP/Data/finetune_checkpoints/$RUN_NAME/." "$RUN_DIR/adapter/"
fi

echo "results:"
ls -lah "$RUN_DIR/results"
echo "adapter:"
ls -lah "$RUN_DIR/adapter"
