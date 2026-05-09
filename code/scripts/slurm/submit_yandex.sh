#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandex_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/yandexgpt_5_lite_8b_lora_iter1/logs

PRE=$(sbatch --parsable scripts/slurm/yandex_pre.sbatch)
echo "yandex pre:        $PRE"

ITER1_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/yandex_iter1_run.sbatch)
echo "yandex iter1 run:  $ITER1_RUN"
ITER1_POST=$(sbatch --parsable --dependency=afterany:$ITER1_RUN scripts/slurm/yandex_iter1_post.sbatch)
echo "yandex iter1 post: $ITER1_POST"

# Когда добавим iter2 / iter3 / cw - дописать сюда соответствующие run + post с
# зависимостью afterok:$PRE (параллельно с iter1 если хватит GPU; иначе SLURM
# сам встанет в очередь)
# ITER2_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/yandex_iter2_run.sbatch)
# ITER2_POST=$(sbatch --parsable --dependency=afterany:$ITER2_RUN scripts/slurm/yandex_iter2_post.sbatch)

squeue -u "$(whoami)"
