#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --job-name=ft_vist_cw_post
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_vistral_24b_qlora_cw/logs/post_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_vistral_24b_qlora_cw/logs/post_%j.err

set -e

RUN_NAME=reaug_vistral_24b_qlora_cw
RUN_DIR=/mnt/pool/6/kvturanosov/VKR/finetune_runs/$RUN_NAME
TMP=/tmp/kvt/finetune_reaug_vistral/$RUN_NAME/code

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

[ -f "$TMP/results/config_used.json" ] && cp -a "$TMP/results/config_used.json" "$RUN_DIR/config_used.json"
[ -f "$TMP/results/manifest.json" ] && cp -a "$TMP/results/manifest.json" "$RUN_DIR/manifest.json"

echo "results:"
ls -lah "$RUN_DIR/results"
echo "adapter:"
ls -lah "$RUN_DIR/adapter"
