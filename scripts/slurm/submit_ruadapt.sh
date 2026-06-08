#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/ruadapt_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/ruadapt_qwen3_32b_qlora_iter1/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/ruadapt_qwen3_32b_adalora_iter1/logs

PRE=$(sbatch --parsable scripts/slurm/ruadapt_pre.sbatch)
echo "ruadapt pre:             $PRE"

QLORA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/ruadapt_qlora_iter1_run.sbatch)
echo "ruadapt qlora   run:     $QLORA_RUN"
QLORA_POST=$(sbatch --parsable --dependency=afterany:$QLORA_RUN scripts/slurm/ruadapt_qlora_iter1_post.sbatch)
echo "ruadapt qlora   post:    $QLORA_POST"

ADA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/ruadapt_adalora_iter1_run.sbatch)
echo "ruadapt adalora run:     $ADA_RUN"
ADA_POST=$(sbatch --parsable --dependency=afterany:$ADA_RUN scripts/slurm/ruadapt_adalora_iter1_post.sbatch)
echo "ruadapt adalora post:    $ADA_POST"

squeue -u "$(whoami)"
