#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/ruadapt_iter3_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/ruadapt_qwen3_32b_qlora_iter3/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/ruadapt_qwen3_32b_adalora_iter3/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/ruadapt_iter3_pre.sbatch)
echo "ruadapt iter3 pre: $PRE"

RUADAPT_QLORA_ITER3_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/ruadapt_qlora_iter3_run.sbatch)
echo "ruadapt_qwen3_32b_qlora_iter3 run:  $RUADAPT_QLORA_ITER3_RUN"
RUADAPT_QLORA_ITER3_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$RUADAPT_QLORA_ITER3_RUN scripts/slurm/ruadapt_qlora_iter3_post.sbatch)
echo "ruadapt_qwen3_32b_qlora_iter3 post: $RUADAPT_QLORA_ITER3_POST"

RUADAPT_ADALORA_ITER3_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/ruadapt_adalora_iter3_run.sbatch)
echo "ruadapt_qwen3_32b_adalora_iter3 run:  $RUADAPT_ADALORA_ITER3_RUN"
RUADAPT_ADALORA_ITER3_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$RUADAPT_ADALORA_ITER3_RUN scripts/slurm/ruadapt_adalora_iter3_post.sbatch)
echo "ruadapt_qwen3_32b_adalora_iter3 post: $RUADAPT_ADALORA_ITER3_POST"

squeue -u "$(whoami)"
