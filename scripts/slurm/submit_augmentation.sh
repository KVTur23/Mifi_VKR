#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code

mkdir -p /mnt/pool/6/kvturanosov/VKR/augmentation_runs/aug_full/logs

PRE=$(sbatch --parsable scripts/slurm/augmentation_pre.sbatch)
echo "aug pre:   $PRE"

RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/augmentation_run.sbatch)
echo "aug run:   $RUN"

POST=$(sbatch --parsable --dependency=afterany:$RUN scripts/slurm/augmentation_post.sbatch)
echo "aug post:  $POST"

squeue -u "$(whoami)"
