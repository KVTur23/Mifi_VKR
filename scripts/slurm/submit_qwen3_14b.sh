#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_14b_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_14b_lora_iter1/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_14b_qlora_iter1/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_14b_adalora_iter1/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_14b_tinylora_iter1/logs

PRE=$(sbatch --parsable scripts/slurm/qwen3_14b_pre.sbatch)
echo "qwen3_14b pre:              $PRE"

LORA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/qwen3_14b_lora_iter1_run.sbatch)
echo "qwen3_14b lora     run:     $LORA_RUN"
LORA_POST=$(sbatch --parsable --dependency=afterany:$LORA_RUN scripts/slurm/qwen3_14b_lora_iter1_post.sbatch)
echo "qwen3_14b lora     post:    $LORA_POST"

QLORA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/qwen3_14b_qlora_iter1_run.sbatch)
echo "qwen3_14b qlora    run:     $QLORA_RUN"
QLORA_POST=$(sbatch --parsable --dependency=afterany:$QLORA_RUN scripts/slurm/qwen3_14b_qlora_iter1_post.sbatch)
echo "qwen3_14b qlora    post:    $QLORA_POST"

ADA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/qwen3_14b_adalora_iter1_run.sbatch)
echo "qwen3_14b adalora  run:     $ADA_RUN"
ADA_POST=$(sbatch --parsable --dependency=afterany:$ADA_RUN scripts/slurm/qwen3_14b_adalora_iter1_post.sbatch)
echo "qwen3_14b adalora  post:    $ADA_POST"

TINY_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/qwen3_14b_tinylora_iter1_run.sbatch)
echo "qwen3_14b tinylora run:     $TINY_RUN"
TINY_POST=$(sbatch --parsable --dependency=afterany:$TINY_RUN scripts/slurm/qwen3_14b_tinylora_iter1_post.sbatch)
echo "qwen3_14b tinylora post:    $TINY_POST"

squeue -u "$(whoami)"
