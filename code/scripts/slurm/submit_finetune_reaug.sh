#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_qwen3_32b_qlora_cw/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_qwen3_32b_qlora_cw_r32/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_qwen3_32b_qlora_cw_focal_g2/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_qwen3_32b_qlora_no_cw_r32/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/reaug_qwen3_32b_qlora_cw_r32_hier_l03/logs

PRE=$(sbatch --parsable scripts/slurm/finetune_reaug_pre.sbatch)
echo "common pre: $PRE"

CW_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/finetune_reaug_cw_run.sbatch)
echo "cw run:     $CW_RUN"
CW_POST=$(sbatch --parsable --dependency=afterany:$CW_RUN scripts/slurm/finetune_reaug_cw_post.sbatch)
echo "cw post:    $CW_POST"

R32_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/finetune_reaug_cw_r32_run.sbatch)
echo "r32 run:    $R32_RUN"
R32_POST=$(sbatch --parsable --dependency=afterany:$R32_RUN scripts/slurm/finetune_reaug_cw_r32_post.sbatch)
echo "r32 post:   $R32_POST"

FOCAL_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/finetune_reaug_cw_focal_g2_run.sbatch)
echo "focal run:  $FOCAL_RUN"
FOCAL_POST=$(sbatch --parsable --dependency=afterany:$FOCAL_RUN scripts/slurm/finetune_reaug_cw_focal_g2_post.sbatch)
echo "focal post: $FOCAL_POST"

NO_CW_R32_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/finetune_reaug_no_cw_r32_run.sbatch)
echo "no cw r32 run:  $NO_CW_R32_RUN"
NO_CW_R32_POST=$(sbatch --parsable --dependency=afterany:$NO_CW_R32_RUN scripts/slurm/finetune_reaug_no_cw_r32_post.sbatch)
echo "no cw r32 post: $NO_CW_R32_POST"

HIER_RUN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/finetune_reaug_cw_r32_hier_l03_run.sbatch)
echo "hier run:  $HIER_RUN"
HIER_POST=$(sbatch --parsable --dependency=afterany:$HIER_RUN scripts/slurm/finetune_reaug_cw_r32_hier_l03_post.sbatch)
echo "hier post: $HIER_POST"

squeue -u "$(whoami)"
