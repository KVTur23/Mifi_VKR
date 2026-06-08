#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_32b_iter4_cw_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_32b_adalora_iter4_cw/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/qwen3_32b_qlora_iter4_cw/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/qwen3_32b_iter4_cw_pre.sbatch)
echo "qwen3_32b iter4 cw pre: $PRE"

QWEN3_32B_ADALORA_ITER4_CW_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/qwen3_32b_adalora_iter4_cw_run.sbatch)
echo "qwen3_32b_adalora_iter4_cw run:  $QWEN3_32B_ADALORA_ITER4_CW_RUN"
QWEN3_32B_ADALORA_ITER4_CW_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$QWEN3_32B_ADALORA_ITER4_CW_RUN scripts/slurm/qwen3_32b_adalora_iter4_cw_post.sbatch)
echo "qwen3_32b_adalora_iter4_cw post: $QWEN3_32B_ADALORA_ITER4_CW_POST"

QWEN3_32B_QLORA_ITER4_CW_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/qwen3_32b_qlora_iter4_cw_run.sbatch)
echo "qwen3_32b_qlora_iter4_cw run:  $QWEN3_32B_QLORA_ITER4_CW_RUN"
QWEN3_32B_QLORA_ITER4_CW_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$QWEN3_32B_QLORA_ITER4_CW_RUN scripts/slurm/qwen3_32b_qlora_iter4_cw_post.sbatch)
echo "qwen3_32b_qlora_iter4_cw post: $QWEN3_32B_QLORA_ITER4_CW_POST"

squeue -u "$(whoami)"
