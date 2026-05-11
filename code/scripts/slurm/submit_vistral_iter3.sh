#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vistral_iter3_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vistral_24b_qlora_iter3/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/vistral_iter3_pre.sbatch)
echo "pre: $PRE"

VISTRAL_ITER3_QLORA_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vistral_qlora_iter3_run.sbatch)
echo "vistral iter3 qlora run:  $VISTRAL_ITER3_QLORA_RUN"
VISTRAL_ITER3_QLORA_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$VISTRAL_ITER3_QLORA_RUN scripts/slurm/vistral_qlora_iter3_post.sbatch)
echo "vistral iter3 qlora post: $VISTRAL_ITER3_QLORA_POST"

squeue -u "$(whoami)"
