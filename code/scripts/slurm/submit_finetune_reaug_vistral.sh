#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_vistral_24b_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_vistral_24b_qlora_cw/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_vistral_24b_qlora_cw_r32/logs

SBATCH_ARGS=${SBATCH_ARGS:-}

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/finetune_reaug_vistral_pre.sbatch)
echo "vistral pre:   $PRE"

CW_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/finetune_reaug_vistral_cw_run.sbatch)
echo "cw run:        $CW_RUN"
CW_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$CW_RUN scripts/slurm/finetune_reaug_vistral_cw_post.sbatch)
echo "cw post:       $CW_POST"

R32_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/finetune_reaug_vistral_cw_r32_run.sbatch)
echo "r32 run:       $R32_RUN"
R32_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$R32_RUN scripts/slurm/finetune_reaug_vistral_cw_r32_post.sbatch)
echo "r32 post:      $R32_POST"

squeue -u "$(whoami)"
