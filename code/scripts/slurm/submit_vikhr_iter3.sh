#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_iter3_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_nemo_12b_lora_iter3/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_nemo_12b_qlora_iter3/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_nemo_12b_adalora_iter3/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_nemo_12b_tinylora_iter3/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/vikhr_iter3_pre.sbatch)
echo "vikhr iter3 pre: $PRE"

VIKHR_LORA_ITER3_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vikhr_lora_iter3_run.sbatch)
echo "vikhr_nemo_12b_lora_iter3 run:  $VIKHR_LORA_ITER3_RUN"
VIKHR_LORA_ITER3_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$VIKHR_LORA_ITER3_RUN scripts/slurm/vikhr_lora_iter3_post.sbatch)
echo "vikhr_nemo_12b_lora_iter3 post: $VIKHR_LORA_ITER3_POST"

VIKHR_QLORA_ITER3_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vikhr_qlora_iter3_run.sbatch)
echo "vikhr_nemo_12b_qlora_iter3 run:  $VIKHR_QLORA_ITER3_RUN"
VIKHR_QLORA_ITER3_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$VIKHR_QLORA_ITER3_RUN scripts/slurm/vikhr_qlora_iter3_post.sbatch)
echo "vikhr_nemo_12b_qlora_iter3 post: $VIKHR_QLORA_ITER3_POST"

VIKHR_ADALORA_ITER3_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vikhr_adalora_iter3_run.sbatch)
echo "vikhr_nemo_12b_adalora_iter3 run:  $VIKHR_ADALORA_ITER3_RUN"
VIKHR_ADALORA_ITER3_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$VIKHR_ADALORA_ITER3_RUN scripts/slurm/vikhr_adalora_iter3_post.sbatch)
echo "vikhr_nemo_12b_adalora_iter3 post: $VIKHR_ADALORA_ITER3_POST"

VIKHR_TINYLORA_ITER3_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vikhr_tinylora_iter3_run.sbatch)
echo "vikhr_nemo_12b_tinylora_iter3 run:  $VIKHR_TINYLORA_ITER3_RUN"
VIKHR_TINYLORA_ITER3_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$VIKHR_TINYLORA_ITER3_RUN scripts/slurm/vikhr_tinylora_iter3_post.sbatch)
echo "vikhr_nemo_12b_tinylora_iter3 post: $VIKHR_TINYLORA_ITER3_POST"

squeue -u "$(whoami)"
