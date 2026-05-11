#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vistral_iter4_cw_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vistral_24b_qlora_iter4_cw/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vistral_24b_adalora_iter4_cw/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/vistral_iter4_cw_pre.sbatch)
echo "vistral iter4 cw pre: $PRE"

VISTRAL_24B_QLORA_ITER4_CW_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vistral_24b_qlora_iter4_cw_run.sbatch)
echo "vistral_24b_qlora_iter4_cw run:  $VISTRAL_24B_QLORA_ITER4_CW_RUN"
VISTRAL_24B_QLORA_ITER4_CW_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$VISTRAL_24B_QLORA_ITER4_CW_RUN scripts/slurm/vistral_24b_qlora_iter4_cw_post.sbatch)
echo "vistral_24b_qlora_iter4_cw post: $VISTRAL_24B_QLORA_ITER4_CW_POST"

VISTRAL_24B_ADALORA_ITER4_CW_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vistral_24b_adalora_iter4_cw_run.sbatch)
echo "vistral_24b_adalora_iter4_cw run:  $VISTRAL_24B_ADALORA_ITER4_CW_RUN"
VISTRAL_24B_ADALORA_ITER4_CW_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$VISTRAL_24B_ADALORA_ITER4_CW_RUN scripts/slurm/vistral_24b_adalora_iter4_cw_post.sbatch)
echo "vistral_24b_adalora_iter4_cw post: $VISTRAL_24B_ADALORA_ITER4_CW_POST"

squeue -u "$(whoami)"
