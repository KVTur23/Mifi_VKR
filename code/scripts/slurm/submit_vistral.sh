#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vistral_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vistral_24b_qlora_iter1/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vistral_24b_adalora_iter1/logs

PRE=$(sbatch --parsable scripts/slurm/vistral_pre.sbatch)
echo "vistral pre:             $PRE"

QLORA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/vistral_qlora_iter1_run.sbatch)
echo "vistral qlora   run:     $QLORA_RUN"
QLORA_POST=$(sbatch --parsable --dependency=afterany:$QLORA_RUN scripts/slurm/vistral_qlora_iter1_post.sbatch)
echo "vistral qlora   post:    $QLORA_POST"

ADA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/vistral_adalora_iter1_run.sbatch)
echo "vistral adalora run:     $ADA_RUN"
ADA_POST=$(sbatch --parsable --dependency=afterany:$ADA_RUN scripts/slurm/vistral_adalora_iter1_post.sbatch)
echo "vistral adalora post:    $ADA_POST"

squeue -u "$(whoami)"
