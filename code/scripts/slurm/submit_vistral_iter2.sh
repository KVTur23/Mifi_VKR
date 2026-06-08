#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vistral_iter2_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vistral_24b_qlora_iter2/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vistral_24b_adalora_iter2/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/vistral_iter2_pre.sbatch)
echo "vistral iter2 pre:              $PRE"

QLORA_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vistral_qlora_iter2_run.sbatch)
echo "vistral iter2 qlora    run:     $QLORA_RUN"
QLORA_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$QLORA_RUN scripts/slurm/vistral_qlora_iter2_post.sbatch)
echo "vistral iter2 qlora    post:    $QLORA_POST"

ADALORA_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vistral_adalora_iter2_run.sbatch)
echo "vistral iter2 adalora  run:     $ADALORA_RUN"
ADALORA_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$ADALORA_RUN scripts/slurm/vistral_adalora_iter2_post.sbatch)
echo "vistral iter2 adalora  post:    $ADALORA_POST"

squeue -u "$(whoami)"
