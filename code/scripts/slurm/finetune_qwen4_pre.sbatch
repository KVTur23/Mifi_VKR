#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --job-name=ft_qw4_pre

set -euo pipefail

: "${BRANCH_KEY:?set BRANCH_KEY, for example test_last or test_sev}"

SRC=${SRC:-/mnt/pool/6/kvturanosov/VKR/worktrees/$BRANCH_KEY/code}
SRC_TREE=$(dirname "$SRC")
BASE_RUN_DIR=/mnt/pool/6/kvturanosov/VKR/finetune_runs
BASE_TMP=/tmp/kvt/finetune_${BRANCH_KEY}_qwen4
MODEL_DIR=models--Qwen--Qwen3-32B
MODEL_MARKER=/tmp/kvt/models/.finetune_qwen3_32b_copy_complete

RUNS="
${BRANCH_KEY}_qwen3_32b_qlora_no_cw
${BRANCH_KEY}_qwen3_32b_qlora_cw_v10
${BRANCH_KEY}_qwen3_32b_qlora_no_cw_r32
${BRANCH_KEY}_qwen3_32b_qlora_cw_v10_r32
"

echo "=== common pre | $BRANCH_KEY qwen4 | $(date) ==="
echo "SRC=$SRC"

mkdir -p /tmp/kvt/models /tmp/kvt/hf /tmp/kvt/cache /tmp/kvt/cache/tmp

model_complete() {
    local root=$1
    [ -d "$root/$MODEL_DIR" ] || return 1
    [ -n "$(find "$root/$MODEL_DIR/snapshots" -name config.json -print -quit 2>/dev/null)" ] || return 1
    [ -n "$(find "$root/$MODEL_DIR/snapshots" -name tokenizer_config.json -print -quit 2>/dev/null)" ] || return 1
    [ -n "$(find "$root/$MODEL_DIR/snapshots" -name model.safetensors -print -quit 2>/dev/null)" ] && return 0
    [ -n "$(find "$root/$MODEL_DIR/snapshots" -name model.safetensors.index.json -print -quit 2>/dev/null)" ] && return 0
    [ -n "$(find "$root/$MODEL_DIR/snapshots" -name pytorch_model.bin -print -quit 2>/dev/null)" ] && return 0
    return 1
}

if [ -f "$MODEL_MARKER" ] && model_complete /tmp/kvt/models; then
    echo "model already copied: $MODEL_DIR"
else
    if ! model_complete /mnt/pool/6/kvturanosov/VKR/models; then
        echo "[ERROR] incomplete model cache: /mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR"
        echo "        expected config/tokenizer plus model.safetensors or model.safetensors.index.json"
        exit 1
    fi

    echo "copy/update model -> /tmp/kvt/models/$MODEL_DIR"
    rm -rf "/tmp/kvt/models/$MODEL_DIR"
    cp -a "/mnt/pool/6/kvturanosov/VKR/models/$MODEL_DIR" /tmp/kvt/models/
    [ -f /mnt/pool/6/kvturanosov/VKR/models/CACHEDIR.TAG ] && cp -a /mnt/pool/6/kvturanosov/VKR/models/CACHEDIR.TAG /tmp/kvt/models/
    date > "$MODEL_MARKER"
fi

for RUN_NAME in $RUNS; do
    RUN_DIR="$BASE_RUN_DIR/$RUN_NAME"
    TMP="$BASE_TMP/$RUN_NAME/code"

    echo
    echo "=== prepare $RUN_NAME ==="
    mkdir -p "$RUN_DIR/logs" "$RUN_DIR/results" "$RUN_DIR/Data"

    echo "copy code -> $TMP"
    rm -rf "$BASE_TMP/$RUN_NAME"
    mkdir -p "$TMP"
    cp -a "$SRC/." "$TMP/"

    rm -rf "$TMP/Data" "$TMP/results" "$TMP/logs"
    mkdir -p "$TMP/Data" "$TMP/results" "$TMP/logs" "$TMP/Data/finetune_checkpoints"

    echo "refresh run Data from branch"
    rm -rf "$RUN_DIR/Data"
    mkdir -p "$RUN_DIR/Data"
    cp -a "$SRC/Data/." "$RUN_DIR/Data/"

    echo "copy run Data -> tmp"
    cp -a "$RUN_DIR/Data/." "$TMP/Data/"

    git -C "$SRC_TREE" rev-parse HEAD > "$TMP/source_commit.txt" 2>/dev/null || true
    echo "data files:"
    ls -lah "$TMP/Data"
done

echo
echo "common pre ready"
