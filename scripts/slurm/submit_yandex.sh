#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandex_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_lora_iter1/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_qlora_iter1/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_adalora_iter1/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_tinylora_iter1/logs

PRE=$(sbatch --parsable scripts/slurm/yandex_pre.sbatch)
echo "yandex pre:              $PRE"

LORA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/yandex_lora_iter1_run.sbatch)
echo "yandex lora     run:     $LORA_RUN"
LORA_POST=$(sbatch --parsable --dependency=afterany:$LORA_RUN scripts/slurm/yandex_lora_iter1_post.sbatch)
echo "yandex lora     post:    $LORA_POST"

QLORA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/yandex_qlora_iter1_run.sbatch)
echo "yandex qlora    run:     $QLORA_RUN"
QLORA_POST=$(sbatch --parsable --dependency=afterany:$QLORA_RUN scripts/slurm/yandex_qlora_iter1_post.sbatch)
echo "yandex qlora    post:    $QLORA_POST"

ADA_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/yandex_adalora_iter1_run.sbatch)
echo "yandex adalora  run:     $ADA_RUN"
ADA_POST=$(sbatch --parsable --dependency=afterany:$ADA_RUN scripts/slurm/yandex_adalora_iter1_post.sbatch)
echo "yandex adalora  post:    $ADA_POST"

TINY_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/yandex_tinylora_iter1_run.sbatch)
echo "yandex tinylora run:     $TINY_RUN"
TINY_POST=$(sbatch --parsable --dependency=afterany:$TINY_RUN scripts/slurm/yandex_tinylora_iter1_post.sbatch)
echo "yandex tinylora post:    $TINY_POST"

squeue -u "$(whoami)"
