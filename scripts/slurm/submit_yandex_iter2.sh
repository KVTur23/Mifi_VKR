#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandex_iter2_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_lora_iter2/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_qlora_iter2/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_adalora_iter2/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_tinylora_iter2/logs

PRE=$(sbatch --parsable scripts/slurm/yandex_iter2_pre.sbatch)
echo "yandex iter2 pre:              $PRE"

LORA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/yandex_lora_iter2_run.sbatch)
echo "yandex iter2 lora     run:     $LORA_RUN"
LORA_POST=$(sbatch --parsable --dependency=afterany:$LORA_RUN scripts/slurm/yandex_lora_iter2_post.sbatch)
echo "yandex iter2 lora     post:    $LORA_POST"

QLORA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/yandex_qlora_iter2_run.sbatch)
echo "yandex iter2 qlora    run:     $QLORA_RUN"
QLORA_POST=$(sbatch --parsable --dependency=afterany:$QLORA_RUN scripts/slurm/yandex_qlora_iter2_post.sbatch)
echo "yandex iter2 qlora    post:    $QLORA_POST"

ADA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/yandex_adalora_iter2_run.sbatch)
echo "yandex iter2 adalora  run:     $ADA_RUN"
ADA_POST=$(sbatch --parsable --dependency=afterany:$ADA_RUN scripts/slurm/yandex_adalora_iter2_post.sbatch)
echo "yandex iter2 adalora  post:    $ADA_POST"

TINY_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/yandex_tinylora_iter2_run.sbatch)
echo "yandex iter2 tinylora run:     $TINY_RUN"
TINY_POST=$(sbatch --parsable --dependency=afterany:$TINY_RUN scripts/slurm/yandex_tinylora_iter2_post.sbatch)
echo "yandex iter2 tinylora post:    $TINY_POST"

squeue -u "$(whoami)"
