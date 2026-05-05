#!/bin/bash
set -e

mkdir -p /mnt/pool/6/kvturanosov/VKR/augmentation_runs/rut5nllb_chunked_augment/logs

cd /mnt/pool/6/kvturanosov/VKR/worktrees/rut5-nllb-chunked-augment/code

PRE=$(sbatch --parsable scripts/slurm/augmentation_pre.sbatch)
echo "pre:  $PRE"

RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/augmentation_run.sbatch)
echo "run:  $RUN"

POST=$(sbatch --parsable --dependency=afterany:$RUN scripts/slurm/augmentation_post.sbatch)
echo "post: $POST"

squeue -u "$(whoami)"
