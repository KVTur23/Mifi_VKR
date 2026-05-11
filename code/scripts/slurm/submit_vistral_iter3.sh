#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vistral_iter3_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vistral_24b_qlora_iter3/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vistral_24b_adalora_iter3/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/vistral_iter3_pre.sbatch)
echo "vistral iter3 pre: $PRE"

VISTRAL_QLORA_ITER3_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vistral_qlora_iter3_run.sbatch)
echo "vistral_24b_qlora_iter3 run:  $VISTRAL_QLORA_ITER3_RUN"
VISTRAL_QLORA_ITER3_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$VISTRAL_QLORA_ITER3_RUN scripts/slurm/vistral_qlora_iter3_post.sbatch)
echo "vistral_24b_qlora_iter3 post: $VISTRAL_QLORA_ITER3_POST"

VISTRAL_ADALORA_ITER3_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vistral_adalora_iter3_run.sbatch)
echo "vistral_24b_adalora_iter3 run:  $VISTRAL_ADALORA_ITER3_RUN"
VISTRAL_ADALORA_ITER3_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$VISTRAL_ADALORA_ITER3_RUN scripts/slurm/vistral_adalora_iter3_post.sbatch)
echo "vistral_24b_adalora_iter3 post: $VISTRAL_ADALORA_ITER3_POST"

squeue -u "$(whoami)"
