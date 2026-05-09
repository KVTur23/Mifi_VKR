#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_nemo_12b_lora_iter1/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_nemo_12b_qlora_iter1/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_nemo_12b_adalora_iter1/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_nemo_12b_tinylora_iter1/logs

PRE=$(sbatch --parsable scripts/slurm/vikhr_pre.sbatch)
echo "vikhr pre:              $PRE"

LORA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/vikhr_lora_iter1_run.sbatch)
echo "vikhr lora     run:     $LORA_RUN"
LORA_POST=$(sbatch --parsable --dependency=afterany:$LORA_RUN scripts/slurm/vikhr_lora_iter1_post.sbatch)
echo "vikhr lora     post:    $LORA_POST"

QLORA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/vikhr_qlora_iter1_run.sbatch)
echo "vikhr qlora    run:     $QLORA_RUN"
QLORA_POST=$(sbatch --parsable --dependency=afterany:$QLORA_RUN scripts/slurm/vikhr_qlora_iter1_post.sbatch)
echo "vikhr qlora    post:    $QLORA_POST"

ADA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/vikhr_adalora_iter1_run.sbatch)
echo "vikhr adalora  run:     $ADA_RUN"
ADA_POST=$(sbatch --parsable --dependency=afterany:$ADA_RUN scripts/slurm/vikhr_adalora_iter1_post.sbatch)
echo "vikhr adalora  post:    $ADA_POST"

TINY_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/vikhr_tinylora_iter1_run.sbatch)
echo "vikhr tinylora run:     $TINY_RUN"
TINY_POST=$(sbatch --parsable --dependency=afterany:$TINY_RUN scripts/slurm/vikhr_tinylora_iter1_post.sbatch)
echo "vikhr tinylora post:    $TINY_POST"

squeue -u "$(whoami)"
