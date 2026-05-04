#!/bin/bash
set -euo pipefail

POOL_ROOT="${POOL_ROOT:-/mnt/pool/6/kvturanosov/VKR}"
AUG_RUN_ROOT="${AUG_RUN_ROOT:-$POOL_ROOT/augmentation_runs}"
WORKTREES_ROOT="${WORKTREES_ROOT:-$POOL_ROOT/worktrees}"
BRANCH="${BRANCH:-rut5-nllb-chunked-augment}"
RUN_NAME="${RUN_NAME:-$BRANCH}"
SOURCE_CODE="${SOURCE_CODE:-}"

if [ -z "$RUN_NAME" ]; then
    echo "RUN_NAME is empty"
    exit 1
fi

RUN_DIR="$AUG_RUN_ROOT/$RUN_NAME"
TMP_ROOT="${TMP_ROOT:-/tmp/kvt}"
TMP_RUN="$TMP_ROOT/augmentation_runs/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
JOB_ID="${SLURM_JOB_ID:-manual}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/pre_${JOB_ID}.log") 2>&1

echo "=== $(date) | augmentation pre | run=$RUN_NAME | branch=$BRANCH ==="
echo "POOL_ROOT=$POOL_ROOT"
echo "RUN_DIR=$RUN_DIR"
echo "TMP_RUN=$TMP_RUN"

resolve_source_code() {
    if [ -n "$SOURCE_CODE" ]; then
        if [ -d "$SOURCE_CODE/scripts" ] && [ -d "$SOURCE_CODE/src" ]; then
            echo "$SOURCE_CODE"
            return 0
        fi
        echo "SOURCE_CODE is set but does not look like code root: $SOURCE_CODE" >&2
        return 1
    fi

    local worktree="$WORKTREES_ROOT/$BRANCH"
    if [ -d "$worktree/code/scripts" ] && [ -d "$worktree/code/src" ]; then
        echo "$worktree/code"
        return 0
    fi
    if [ -d "$worktree/scripts" ] && [ -d "$worktree/src" ]; then
        echo "$worktree"
        return 0
    fi
    if [ -d "$POOL_ROOT/code/scripts" ] && [ -d "$POOL_ROOT/code/src" ]; then
        echo "$POOL_ROOT/code"
        return 0
    fi

    echo "Cannot resolve source code for branch=$BRANCH under $WORKTREES_ROOT" >&2
    return 1
}

SRC_CODE="$(resolve_source_code)"
echo "SRC_CODE=$SRC_CODE"

mkdir -p "$TMP_ROOT/models" "$TMP_ROOT/hf" "$TMP_ROOT/cache" "$RUN_DIR/Data" "$RUN_DIR/results"

if [ -d "$POOL_ROOT/models" ]; then
    if [ -f "$TMP_ROOT/models/.copy_complete" ]; then
        echo "Models already copied to $TMP_ROOT/models, skip"
    else
        echo "Copying models: $POOL_ROOT/models -> $TMP_ROOT/models"
        rsync -a --info=progress2 "$POOL_ROOT/models/" "$TMP_ROOT/models/"
        date > "$TMP_ROOT/models/.copy_complete"
        echo "Models size: $(du -sh "$TMP_ROOT/models" | cut -f1)"
    fi
else
    echo "[WARN] $POOL_ROOT/models does not exist; relying on existing HF cache/offline paths"
fi

if [ -z "$(find "$RUN_DIR/Data" -mindepth 1 -maxdepth 1 2>/dev/null | head -1)" ]; then
    echo "Seeding run Data from $SRC_CODE/Data"
    rsync -a "$SRC_CODE/Data/" "$RUN_DIR/Data/"
else
    echo "Run Data already exists: $RUN_DIR/Data"
fi

echo "Preparing tmp code at $TMP_RUN/code"
rm -rf "$TMP_RUN"
mkdir -p "$TMP_RUN/code"

rsync -a \
    --exclude '/Data' \
    --exclude '/results' \
    --exclude '/logs' \
    --exclude '/__pycache__' \
    "$SRC_CODE/" "$TMP_RUN/code/"

mkdir -p "$TMP_RUN/code/Data" "$TMP_RUN/code/results" "$TMP_RUN/code/logs"
rsync -a "$RUN_DIR/Data/" "$TMP_RUN/code/Data/"
if [ -d "$RUN_DIR/results" ]; then
    rsync -a "$RUN_DIR/results/" "$TMP_RUN/code/results/"
fi

if git -C "$SRC_CODE" rev-parse HEAD >/dev/null 2>&1; then
    git -C "$SRC_CODE" rev-parse HEAD > "$TMP_RUN/code/source_commit.txt"
    git -C "$SRC_CODE" log -1 --oneline > "$RUN_DIR/source_commit.txt"
fi

echo "Tmp Data files:"
find "$TMP_RUN/code/Data" -maxdepth 1 -type f -printf "  %f\n" | sort || true
echo "Ready: $TMP_RUN/code"
