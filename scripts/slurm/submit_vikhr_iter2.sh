#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_iter2_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_nemo_12b_lora_iter2/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_nemo_12b_qlora_iter2/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_nemo_12b_adalora_iter2/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_nemo_12b_tinylora_iter2/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/vikhr_iter2_pre.sbatch)
echo "vikhr iter2 pre:              $PRE"

LORA_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vikhr_lora_iter2_run.sbatch)
echo "vikhr iter2 lora     run:     $LORA_RUN"
LORA_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$LORA_RUN scripts/slurm/vikhr_lora_iter2_post.sbatch)
echo "vikhr iter2 lora     post:    $LORA_POST"

QLORA_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vikhr_qlora_iter2_run.sbatch)
echo "vikhr iter2 qlora    run:     $QLORA_RUN"
QLORA_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$QLORA_RUN scripts/slurm/vikhr_qlora_iter2_post.sbatch)
echo "vikhr iter2 qlora    post:    $QLORA_POST"

ADALORA_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vikhr_adalora_iter2_run.sbatch)
echo "vikhr iter2 adalora  run:     $ADALORA_RUN"
ADALORA_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$ADALORA_RUN scripts/slurm/vikhr_adalora_iter2_post.sbatch)
echo "vikhr iter2 adalora  post:    $ADALORA_POST"

TINYLORA_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vikhr_tinylora_iter2_run.sbatch)
echo "vikhr iter2 tinylora run:     $TINYLORA_RUN"
TINYLORA_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$TINYLORA_RUN scripts/slurm/vikhr_tinylora_iter2_post.sbatch)
echo "vikhr iter2 tinylora post:    $TINYLORA_POST"

squeue -u "$(whoami)"
