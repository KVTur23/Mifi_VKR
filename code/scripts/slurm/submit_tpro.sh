#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/tpro_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/tpro_it_21_qlora_cw/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/tpro_it_21_adalora_cw/logs

PRE=$(sbatch --parsable scripts/slurm/tpro_pre.sbatch)
echo "tpro pre:                  $PRE"

QLORA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/tpro_qlora_cw_run.sbatch)
echo "tpro qlora_cw   run:       $QLORA_RUN"
QLORA_POST=$(sbatch --parsable --dependency=afterany:$QLORA_RUN scripts/slurm/tpro_qlora_cw_post.sbatch)
echo "tpro qlora_cw   post:      $QLORA_POST"

ADA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/tpro_adalora_cw_run.sbatch)
echo "tpro adalora_cw run:       $ADA_RUN"
ADA_POST=$(sbatch --parsable --dependency=afterany:$ADA_RUN scripts/slurm/tpro_adalora_cw_post.sbatch)
echo "tpro adalora_cw post:      $ADA_POST"

squeue -u "$(whoami)"
