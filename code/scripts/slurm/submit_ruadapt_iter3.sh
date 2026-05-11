#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/ruadapt_iter3_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/ruadapt_qwen3_32b_qlora_iter3/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/ruadapt_iter3_pre.sbatch)
echo "pre: $PRE"

RUADAPT_ITER3_QLORA_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/ruadapt_qlora_iter3_run.sbatch)
echo "ruadapt iter3 qlora run:  $RUADAPT_ITER3_QLORA_RUN"
RUADAPT_ITER3_QLORA_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$RUADAPT_ITER3_QLORA_RUN scripts/slurm/ruadapt_qlora_iter3_post.sbatch)
echo "ruadapt iter3 qlora post: $RUADAPT_ITER3_QLORA_POST"

squeue -u "$(whoami)"
