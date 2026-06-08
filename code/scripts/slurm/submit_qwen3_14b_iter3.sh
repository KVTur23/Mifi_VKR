#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_14b_iter3_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_14b_lora_iter3/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_14b_qlora_iter3/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_14b_adalora_iter3/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_14b_tinylora_iter3/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/qwen3_14b_iter3_pre.sbatch)
echo "qwen3_14b iter3 pre: $PRE"

QWEN3_14B_LORA_ITER3_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/qwen3_14b_lora_iter3_run.sbatch)
echo "qwen3_14b_lora_iter3 run:  $QWEN3_14B_LORA_ITER3_RUN"
QWEN3_14B_LORA_ITER3_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$QWEN3_14B_LORA_ITER3_RUN scripts/slurm/qwen3_14b_lora_iter3_post.sbatch)
echo "qwen3_14b_lora_iter3 post: $QWEN3_14B_LORA_ITER3_POST"

QWEN3_14B_QLORA_ITER3_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/qwen3_14b_qlora_iter3_run.sbatch)
echo "qwen3_14b_qlora_iter3 run:  $QWEN3_14B_QLORA_ITER3_RUN"
QWEN3_14B_QLORA_ITER3_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$QWEN3_14B_QLORA_ITER3_RUN scripts/slurm/qwen3_14b_qlora_iter3_post.sbatch)
echo "qwen3_14b_qlora_iter3 post: $QWEN3_14B_QLORA_ITER3_POST"

QWEN3_14B_ADALORA_ITER3_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/qwen3_14b_adalora_iter3_run.sbatch)
echo "qwen3_14b_adalora_iter3 run:  $QWEN3_14B_ADALORA_ITER3_RUN"
QWEN3_14B_ADALORA_ITER3_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$QWEN3_14B_ADALORA_ITER3_RUN scripts/slurm/qwen3_14b_adalora_iter3_post.sbatch)
echo "qwen3_14b_adalora_iter3 post: $QWEN3_14B_ADALORA_ITER3_POST"

QWEN3_14B_TINYLORA_ITER3_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/qwen3_14b_tinylora_iter3_run.sbatch)
echo "qwen3_14b_tinylora_iter3 run:  $QWEN3_14B_TINYLORA_ITER3_RUN"
QWEN3_14B_TINYLORA_ITER3_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$QWEN3_14B_TINYLORA_ITER3_RUN scripts/slurm/qwen3_14b_tinylora_iter3_post.sbatch)
echo "qwen3_14b_tinylora_iter3 post: $QWEN3_14B_TINYLORA_ITER3_POST"

squeue -u "$(whoami)"
