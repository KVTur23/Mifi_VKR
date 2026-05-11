#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandex_best_cw_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_qlora_best_cw/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/yandex_best_cw_pre.sbatch)
echo "pre: $PRE"

YANDEX_BESTCW_QLORA_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/yandex_qlora_best_cw_run.sbatch)
echo "yandex best+cw qlora run:  $YANDEX_BESTCW_QLORA_RUN"
YANDEX_BESTCW_QLORA_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$YANDEX_BESTCW_QLORA_RUN scripts/slurm/yandex_qlora_best_cw_post.sbatch)
echo "yandex best+cw qlora post: $YANDEX_BESTCW_QLORA_POST"

squeue -u "$(whoami)"
