#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code
SBATCH_ARGS=${SBATCH_ARGS:-}

SBATCH_ARGS="$SBATCH_ARGS" bash scripts/slurm/submit_yandex_best_cw.sh
SBATCH_ARGS="$SBATCH_ARGS" bash scripts/slurm/submit_qwen3_14b_iter3.sh
SBATCH_ARGS="$SBATCH_ARGS" bash scripts/slurm/submit_qwen25_32b_iter3.sh
SBATCH_ARGS="$SBATCH_ARGS" bash scripts/slurm/submit_ruadapt_iter3.sh
SBATCH_ARGS="$SBATCH_ARGS" bash scripts/slurm/submit_vistral_iter3.sh
SBATCH_ARGS="$SBATCH_ARGS" bash scripts/slurm/submit_vikhr_iter3.sh
