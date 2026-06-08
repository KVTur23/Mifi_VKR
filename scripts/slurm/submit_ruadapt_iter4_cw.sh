#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/ruadapt_iter4_cw_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/ruadapt_qwen3_32b_qlora_iter4_cw/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/ruadapt_qwen3_32b_adalora_iter4_cw/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/ruadapt_iter4_cw_pre.sbatch)
echo "ruadapt iter4 cw pre: $PRE"

RUADAPT_QWEN3_32B_QLORA_ITER4_CW_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/ruadapt_qwen3_32b_qlora_iter4_cw_run.sbatch)
echo "ruadapt_qwen3_32b_qlora_iter4_cw run:  $RUADAPT_QWEN3_32B_QLORA_ITER4_CW_RUN"
RUADAPT_QWEN3_32B_QLORA_ITER4_CW_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$RUADAPT_QWEN3_32B_QLORA_ITER4_CW_RUN scripts/slurm/ruadapt_qwen3_32b_qlora_iter4_cw_post.sbatch)
echo "ruadapt_qwen3_32b_qlora_iter4_cw post: $RUADAPT_QWEN3_32B_QLORA_ITER4_CW_POST"

RUADAPT_QWEN3_32B_ADALORA_ITER4_CW_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/ruadapt_qwen3_32b_adalora_iter4_cw_run.sbatch)
echo "ruadapt_qwen3_32b_adalora_iter4_cw run:  $RUADAPT_QWEN3_32B_ADALORA_ITER4_CW_RUN"
RUADAPT_QWEN3_32B_ADALORA_ITER4_CW_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$RUADAPT_QWEN3_32B_ADALORA_ITER4_CW_RUN scripts/slurm/ruadapt_qwen3_32b_adalora_iter4_cw_post.sbatch)
echo "ruadapt_qwen3_32b_adalora_iter4_cw post: $RUADAPT_QWEN3_32B_ADALORA_ITER4_CW_POST"

squeue -u "$(whoami)"
