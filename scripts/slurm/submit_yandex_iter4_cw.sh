#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandex_iter4_cw_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_lora_iter4_cw/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_qlora_iter4_cw/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_adalora_iter4_cw/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_tinylora_iter4_cw/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/yandex_iter4_cw_pre.sbatch)
echo "yandex iter4 cw pre: $PRE"

YANDEXGPT_5_LITE_8B_LORA_ITER4_CW_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/yandexgpt_5_lite_8b_lora_iter4_cw_run.sbatch)
echo "yandexgpt_5_lite_8b_lora_iter4_cw run:  $YANDEXGPT_5_LITE_8B_LORA_ITER4_CW_RUN"
YANDEXGPT_5_LITE_8B_LORA_ITER4_CW_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$YANDEXGPT_5_LITE_8B_LORA_ITER4_CW_RUN scripts/slurm/yandexgpt_5_lite_8b_lora_iter4_cw_post.sbatch)
echo "yandexgpt_5_lite_8b_lora_iter4_cw post: $YANDEXGPT_5_LITE_8B_LORA_ITER4_CW_POST"

YANDEXGPT_5_LITE_8B_QLORA_ITER4_CW_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/yandexgpt_5_lite_8b_qlora_iter4_cw_run.sbatch)
echo "yandexgpt_5_lite_8b_qlora_iter4_cw run:  $YANDEXGPT_5_LITE_8B_QLORA_ITER4_CW_RUN"
YANDEXGPT_5_LITE_8B_QLORA_ITER4_CW_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$YANDEXGPT_5_LITE_8B_QLORA_ITER4_CW_RUN scripts/slurm/yandexgpt_5_lite_8b_qlora_iter4_cw_post.sbatch)
echo "yandexgpt_5_lite_8b_qlora_iter4_cw post: $YANDEXGPT_5_LITE_8B_QLORA_ITER4_CW_POST"

YANDEXGPT_5_LITE_8B_ADALORA_ITER4_CW_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/yandexgpt_5_lite_8b_adalora_iter4_cw_run.sbatch)
echo "yandexgpt_5_lite_8b_adalora_iter4_cw run:  $YANDEXGPT_5_LITE_8B_ADALORA_ITER4_CW_RUN"
YANDEXGPT_5_LITE_8B_ADALORA_ITER4_CW_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$YANDEXGPT_5_LITE_8B_ADALORA_ITER4_CW_RUN scripts/slurm/yandexgpt_5_lite_8b_adalora_iter4_cw_post.sbatch)
echo "yandexgpt_5_lite_8b_adalora_iter4_cw post: $YANDEXGPT_5_LITE_8B_ADALORA_ITER4_CW_POST"

YANDEXGPT_5_LITE_8B_TINYLORA_ITER4_CW_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/yandexgpt_5_lite_8b_tinylora_iter4_cw_run.sbatch)
echo "yandexgpt_5_lite_8b_tinylora_iter4_cw run:  $YANDEXGPT_5_LITE_8B_TINYLORA_ITER4_CW_RUN"
YANDEXGPT_5_LITE_8B_TINYLORA_ITER4_CW_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$YANDEXGPT_5_LITE_8B_TINYLORA_ITER4_CW_RUN scripts/slurm/yandexgpt_5_lite_8b_tinylora_iter4_cw_post.sbatch)
echo "yandexgpt_5_lite_8b_tinylora_iter4_cw post: $YANDEXGPT_5_LITE_8B_TINYLORA_ITER4_CW_POST"

squeue -u "$(whoami)"
