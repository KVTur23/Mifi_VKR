#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_14b_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_14b_lora_iter1/logs

PRE=$(sbatch --parsable scripts/slurm/qwen3_14b_pre.sbatch)
echo "qwen3_14b pre:        $PRE"

ITER1_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/qwen3_14b_iter1_run.sbatch)
echo "qwen3_14b iter1 run:  $ITER1_RUN"
ITER1_POST=$(sbatch --parsable --dependency=afterany:$ITER1_RUN scripts/slurm/qwen3_14b_iter1_post.sbatch)
echo "qwen3_14b iter1 post: $ITER1_POST"

squeue -u "$(whoami)"
