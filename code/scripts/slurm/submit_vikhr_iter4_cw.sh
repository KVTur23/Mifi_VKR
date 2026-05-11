#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_iter4_cw_pre/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_nemo_12b_lora_iter4_cw/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_nemo_12b_qlora_iter4_cw/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_nemo_12b_adalora_iter4_cw/logs
mkdir -p /mnt/pool/6/kvturanosov/VKR/finetune_runs/vikhr_nemo_12b_tinylora_iter4_cw/logs

PRE=$(sbatch --parsable $SBATCH_ARGS scripts/slurm/vikhr_iter4_cw_pre.sbatch)
echo "vikhr iter4 cw pre: $PRE"

VIKHR_NEMO_12B_LORA_ITER4_CW_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vikhr_nemo_12b_lora_iter4_cw_run.sbatch)
echo "vikhr_nemo_12b_lora_iter4_cw run:  $VIKHR_NEMO_12B_LORA_ITER4_CW_RUN"
VIKHR_NEMO_12B_LORA_ITER4_CW_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$VIKHR_NEMO_12B_LORA_ITER4_CW_RUN scripts/slurm/vikhr_nemo_12b_lora_iter4_cw_post.sbatch)
echo "vikhr_nemo_12b_lora_iter4_cw post: $VIKHR_NEMO_12B_LORA_ITER4_CW_POST"

VIKHR_NEMO_12B_QLORA_ITER4_CW_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vikhr_nemo_12b_qlora_iter4_cw_run.sbatch)
echo "vikhr_nemo_12b_qlora_iter4_cw run:  $VIKHR_NEMO_12B_QLORA_ITER4_CW_RUN"
VIKHR_NEMO_12B_QLORA_ITER4_CW_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$VIKHR_NEMO_12B_QLORA_ITER4_CW_RUN scripts/slurm/vikhr_nemo_12b_qlora_iter4_cw_post.sbatch)
echo "vikhr_nemo_12b_qlora_iter4_cw post: $VIKHR_NEMO_12B_QLORA_ITER4_CW_POST"

VIKHR_NEMO_12B_ADALORA_ITER4_CW_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vikhr_nemo_12b_adalora_iter4_cw_run.sbatch)
echo "vikhr_nemo_12b_adalora_iter4_cw run:  $VIKHR_NEMO_12B_ADALORA_ITER4_CW_RUN"
VIKHR_NEMO_12B_ADALORA_ITER4_CW_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$VIKHR_NEMO_12B_ADALORA_ITER4_CW_RUN scripts/slurm/vikhr_nemo_12b_adalora_iter4_cw_post.sbatch)
echo "vikhr_nemo_12b_adalora_iter4_cw post: $VIKHR_NEMO_12B_ADALORA_ITER4_CW_POST"

VIKHR_NEMO_12B_TINYLORA_ITER4_CW_RUN=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$PRE scripts/slurm/vikhr_nemo_12b_tinylora_iter4_cw_run.sbatch)
echo "vikhr_nemo_12b_tinylora_iter4_cw run:  $VIKHR_NEMO_12B_TINYLORA_ITER4_CW_RUN"
VIKHR_NEMO_12B_TINYLORA_ITER4_CW_POST=$(sbatch --parsable $SBATCH_ARGS --dependency=afterany:$VIKHR_NEMO_12B_TINYLORA_ITER4_CW_RUN scripts/slurm/vikhr_nemo_12b_tinylora_iter4_cw_post.sbatch)
echo "vikhr_nemo_12b_tinylora_iter4_cw post: $VIKHR_NEMO_12B_TINYLORA_ITER4_CW_POST"

squeue -u "$(whoami)"
