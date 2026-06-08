#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_32b_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_32b_adalora_cw/logs

PRE=$(sbatch --parsable scripts/slurm/qwen3_32b_pre.sbatch)
echo "qwen3_32b pre:               $PRE"

RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/qwen3_32b_adalora_cw_run.sbatch)
echo "qwen3_32b adalora_cw run:    $RUN"
POST=$(sbatch --parsable --dependency=afterany:$RUN scripts/slurm/qwen3_32b_adalora_cw_post.sbatch)
echo "qwen3_32b adalora_cw post:   $POST"

squeue -u "$(whoami)"
