#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen25_32b_iter3_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen25_32b_qlora_iter3/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen25_32b_adalora_iter3/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/qwen25_32b_iter3_pre.sbatch)
echo "qwen25_32b iter3 pre: $PRE"

QWEN25_32B_QLORA_ITER3_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/qwen25_32b_qlora_iter3_run.sbatch)
echo "qwen25_32b_qlora_iter3 run:  $QWEN25_32B_QLORA_ITER3_RUN"
QWEN25_32B_QLORA_ITER3_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$QWEN25_32B_QLORA_ITER3_RUN scripts/slurm/qwen25_32b_qlora_iter3_post.sbatch)
echo "qwen25_32b_qlora_iter3 post: $QWEN25_32B_QLORA_ITER3_POST"

QWEN25_32B_ADALORA_ITER3_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/qwen25_32b_adalora_iter3_run.sbatch)
echo "qwen25_32b_adalora_iter3 run:  $QWEN25_32B_ADALORA_ITER3_RUN"
QWEN25_32B_ADALORA_ITER3_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$QWEN25_32B_ADALORA_ITER3_RUN scripts/slurm/qwen25_32b_adalora_iter3_post.sbatch)
echo "qwen25_32b_adalora_iter3 post: $QWEN25_32B_ADALORA_ITER3_POST"

squeue -u "$(whoami)"
