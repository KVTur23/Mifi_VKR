#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen25_32b_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen25_32b_qlora_iter1/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen25_32b_adalora_iter1/logs

PRE=$(sbatch --parsable scripts/slurm/qwen25_32b_pre.sbatch)
echo "qwen25_32b pre:             $PRE"

QLORA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/qwen25_32b_qlora_iter1_run.sbatch)
echo "qwen25_32b qlora   run:     $QLORA_RUN"
QLORA_POST=$(sbatch --parsable --dependency=afterany:$QLORA_RUN scripts/slurm/qwen25_32b_qlora_iter1_post.sbatch)
echo "qwen25_32b qlora   post:    $QLORA_POST"

ADA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/qwen25_32b_adalora_iter1_run.sbatch)
echo "qwen25_32b adalora run:     $ADA_RUN"
ADA_POST=$(sbatch --parsable --dependency=afterany:$ADA_RUN scripts/slurm/qwen25_32b_adalora_iter1_post.sbatch)
echo "qwen25_32b adalora post:    $ADA_POST"

squeue -u "$(whoami)"
