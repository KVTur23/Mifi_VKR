#!/bin/bash
# Submit the augmentation chain: pre -> run -> post.
#
# Usage:
#   bash scripts/slurm/submit_augmentation.sh
#   bash scripts/slurm/submit_augmentation.sh rut5nllb_chunked rut5-nllb-chunked-augment A100_40
#
# Optional env:
#   CONFIG_REL=config_models/aug_configs/model_vllm_32b.json
#   AUG_EXTRA_ARGS="--metrics classical"
#   AUG_RUN_ROOT=/mnt/pool/6/kvturanosov/VKR/augmentation_runs

set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_NAME="${1:-rut5nllb_chunked_augment}"
BRANCH="${2:-rut5-nllb-chunked-augment}"
GPU="${3:-A100_40}"
POOL_ROOT="${POOL_ROOT:-/mnt/pool/6/kvturanosov/VKR}"
CONFIG_REL="${CONFIG_REL:-config_models/aug_configs/model_vllm_32b.json}"
AUG_EXTRA_ARGS="${AUG_EXTRA_ARGS:-}"
AUG_RUN_ROOT="${AUG_RUN_ROOT:-$POOL_ROOT/augmentation_runs}"

echo "Run name : $RUN_NAME"
echo "Branch   : $BRANCH"
echo "GPU      : $GPU"
echo "Config   : $CONFIG_REL"
echo "Run root : $AUG_RUN_ROOT"
if [ -n "$AUG_EXTRA_ARGS" ]; then
    echo "Extra    : $AUG_EXTRA_ARGS"
fi
echo

mkdir -p "$POOL_ROOT/code/logs" "$AUG_RUN_ROOT/$RUN_NAME/logs"
export RUN_NAME BRANCH GPU CONFIG_REL AUG_RUN_ROOT AUG_EXTRA_ARGS

J_PRE=$(sbatch --parsable --export=ALL "$SLURM_DIR/augmentation_pre.sbatch")
echo "pre  : $J_PRE"

J_RUN=$(sbatch --parsable --dependency=afterok:$J_PRE --export=ALL "$SLURM_DIR/augmentation_run.sbatch")
echo "run  : $J_RUN  (afterok:$J_PRE)"

J_POST=$(sbatch --parsable --dependency=afterany:$J_RUN --export=ALL "$SLURM_DIR/augmentation_post.sbatch")
echo "post : $J_POST (afterany:$J_RUN)"

echo
echo "Logs/results:"
echo "  $AUG_RUN_ROOT/$RUN_NAME/logs"
echo "  $AUG_RUN_ROOT/$RUN_NAME/results"
echo "  $AUG_RUN_ROOT/$RUN_NAME/Data"
echo
echo "=== squeue ==="
squeue -u "$(whoami)"
