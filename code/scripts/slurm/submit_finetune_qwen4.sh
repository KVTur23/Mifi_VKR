#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CODE_DIR=$(pwd)
if [ -z "${BRANCH_KEY:-}" ]; then
    PARENT=$(basename "$(dirname "$CODE_DIR")")
    GRANDPARENT=$(basename "$(dirname "$(dirname "$CODE_DIR")")")
    case "$PARENT" in
        test_last|test_sev) BRANCH_KEY=$PARENT ;;
        code) BRANCH_KEY=$GRANDPARENT ;;
        *) BRANCH_KEY=$PARENT ;;
    esac
fi
BASE_RUN_DIR=/mnt/pool/6/kvturanosov/VKR/finetune_runs
GPU_PROFILE=${GPU_PROFILE:-A100_40}

case "$BRANCH_KEY" in
    test_last|test_sev) ;;
    *)
        echo "[ERROR] Cannot infer branch key from $CODE_DIR"
        echo "        Run from /mnt/pool/6/kvturanosov/VKR/worktrees/test_last/code"
        echo "        or set BRANCH_KEY=test_last / BRANCH_KEY=test_sev"
        exit 1
        ;;
esac

read -r -a EXTRA_SBATCH_ARGS <<< "${SBATCH_ARGS:-}"

RUN_NAMES=(
    "${BRANCH_KEY}_qwen3_32b_qlora_no_cw"
    "${BRANCH_KEY}_qwen3_32b_qlora_cw_v10"
    "${BRANCH_KEY}_qwen3_32b_qlora_no_cw_r32"
    "${BRANCH_KEY}_qwen3_32b_qlora_cw_v10_r32"
)

EXPERIMENTS=(
    "no_cw"
    "cw_v10"
    "no_cw_r32"
    "cw_v10_r32"
)

JOB_SUFFIXES=(
    "ncw"
    "cw10"
    "ncw32"
    "cw1032"
)

BRANCH_SHORT=${BRANCH_KEY#test_}
PRE_DIR="$BASE_RUN_DIR/${BRANCH_KEY}_qwen4_pre"
mkdir -p "$PRE_DIR/logs"

for RUN_NAME in "${RUN_NAMES[@]}"; do
    mkdir -p "$BASE_RUN_DIR/$RUN_NAME/logs"
done

PRE=$(sbatch --parsable "${EXTRA_SBATCH_ARGS[@]}" \
    --export=ALL,BRANCH_KEY="$BRANCH_KEY",SRC="$CODE_DIR" \
    --job-name="ft_${BRANCH_SHORT}_pre" \
    --output="$PRE_DIR/logs/pre_%j.log" \
    --error="$PRE_DIR/logs/pre_%j.err" \
    scripts/slurm/finetune_qwen4_pre.sbatch)
echo "common pre: $PRE"

for i in "${!RUN_NAMES[@]}"; do
    RUN_NAME=${RUN_NAMES[$i]}
    EXPERIMENT=${EXPERIMENTS[$i]}
    JOB_SUFFIX=${JOB_SUFFIXES[$i]}
    RUN_DIR="$BASE_RUN_DIR/$RUN_NAME"

    RUN=$(sbatch --parsable "${EXTRA_SBATCH_ARGS[@]}" \
        --dependency=afterok:$PRE \
        --export=ALL,BRANCH_KEY="$BRANCH_KEY",RUN_NAME="$RUN_NAME",EXPERIMENT="$EXPERIMENT",GPU_PROFILE="$GPU_PROFILE" \
        --job-name="ft_${BRANCH_SHORT}_${JOB_SUFFIX}" \
        --output="$RUN_DIR/logs/run_%j.log" \
        --error="$RUN_DIR/logs/run_%j.err" \
        scripts/slurm/finetune_qwen4_run.sbatch)
    echo "$RUN_NAME run:  $RUN"

    POST=$(sbatch --parsable "${EXTRA_SBATCH_ARGS[@]}" \
        --dependency=afterany:$RUN \
        --export=ALL,BRANCH_KEY="$BRANCH_KEY",RUN_NAME="$RUN_NAME" \
        --job-name="fp_${BRANCH_SHORT}_${JOB_SUFFIX}" \
        --output="$RUN_DIR/logs/post_%j.log" \
        --error="$RUN_DIR/logs/post_%j.err" \
        scripts/slurm/finetune_qwen4_post.sbatch)
    echo "$RUN_NAME post: $POST"
done

squeue -u "$(whoami)"
