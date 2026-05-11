#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_14b_iter3_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_14b_qlora_iter3/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/qwen3_14b_iter3_pre.sbatch)
echo "pre: $PRE"

QWEN3_14B_ITER3_QLORA_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/qwen3_14b_qlora_iter3_run.sbatch)
echo "qwen3 14b iter3 qlora run:  $QWEN3_14B_ITER3_QLORA_RUN"
QWEN3_14B_ITER3_QLORA_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$QWEN3_14B_ITER3_QLORA_RUN scripts/slurm/qwen3_14b_qlora_iter3_post.sbatch)
echo "qwen3 14b iter3 qlora post: $QWEN3_14B_ITER3_QLORA_POST"

squeue -u "$(whoami)"
