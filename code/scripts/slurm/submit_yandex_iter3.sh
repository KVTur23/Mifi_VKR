#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandex_iter3_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_lora_iter3/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_qlora_iter3/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_adalora_iter3/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_tinylora_iter3/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/yandex_iter3_pre.sbatch)
echo "yandex iter3 pre:              $PRE"

LORA_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/yandex_lora_iter3_run.sbatch)
echo "yandex iter3 lora     run:     $LORA_RUN"
LORA_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$LORA_RUN scripts/slurm/yandex_lora_iter3_post.sbatch)
echo "yandex iter3 lora     post:    $LORA_POST"

QLORA_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/yandex_qlora_iter3_run.sbatch)
echo "yandex iter3 qlora    run:     $QLORA_RUN"
QLORA_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$QLORA_RUN scripts/slurm/yandex_qlora_iter3_post.sbatch)
echo "yandex iter3 qlora    post:    $QLORA_POST"

ADALORA_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/yandex_adalora_iter3_run.sbatch)
echo "yandex iter3 adalora  run:     $ADALORA_RUN"
ADALORA_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$ADALORA_RUN scripts/slurm/yandex_adalora_iter3_post.sbatch)
echo "yandex iter3 adalora  post:    $ADALORA_POST"

TINYLORA_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/yandex_tinylora_iter3_run.sbatch)
echo "yandex iter3 tinylora run:     $TINYLORA_RUN"
TINYLORA_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$TINYLORA_RUN scripts/slurm/yandex_tinylora_iter3_post.sbatch)
echo "yandex iter3 tinylora post:    $TINYLORA_POST"

squeue -u "$(whoami)"
