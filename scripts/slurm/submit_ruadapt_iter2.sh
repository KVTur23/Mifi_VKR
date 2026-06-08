#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/ruadapt_iter2_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/ruadapt_qwen3_32b_qlora_iter2/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/ruadapt_qwen3_32b_adalora_iter2/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/ruadapt_iter2_pre.sbatch)
echo "ruadapt iter2 pre:              $PRE"

QLORA_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/ruadapt_qlora_iter2_run.sbatch)
echo "ruadapt iter2 qlora    run:     $QLORA_RUN"
QLORA_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$QLORA_RUN scripts/slurm/ruadapt_qlora_iter2_post.sbatch)
echo "ruadapt iter2 qlora    post:    $QLORA_POST"

ADALORA_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/ruadapt_adalora_iter2_run.sbatch)
echo "ruadapt iter2 adalora  run:     $ADALORA_RUN"
ADALORA_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$ADALORA_RUN scripts/slurm/ruadapt_adalora_iter2_post.sbatch)
echo "ruadapt iter2 adalora  post:    $ADALORA_POST"

squeue -u "$(whoami)"
