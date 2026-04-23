#!/bin/bash
# submit_finetune.sh — сабмит всей цепочки: pre → 4×comp → post через --dependency
#
# Запуск:
#     bash scripts/slurm/submit_finetune.sh            # все 4 метода
#     bash scripts/slurm/submit_finetune.sh lora qlora # только выбранные
#
# Выставь этот скрипт на исполнение: chmod +x scripts/slurm/submit_finetune.sh

set -e

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHODS="${@:-lora qlora adalora tinylora}"

echo "Методы: $METHODS"
echo

J_PRE=$(sbatch --parsable "$SLURM_DIR/finetune_pre.sbatch")
echo "pre     : $J_PRE"

COMP_IDS=""
for M in $METHODS; do
    SBATCH_FILE="$SLURM_DIR/finetune_${M}.sbatch"
    if [ ! -f "$SBATCH_FILE" ]; then
        echo "[WARN] $SBATCH_FILE не найден, пропускаю"
        continue
    fi
    J=$(sbatch --parsable --dependency=afterok:$J_PRE "$SBATCH_FILE")
    echo "$M     : $J  (после pre=$J_PRE)"
    COMP_IDS="${COMP_IDS}:${J}"
done

# afterany — post заберёт результаты даже если какие-то методы упали
DEP="afterany${COMP_IDS}"
J_POST=$(sbatch --parsable --dependency=$DEP "$SLURM_DIR/finetune_post.sbatch")
echo "post    : $J_POST  (после $DEP)"

echo
echo "=== squeue ==="
squeue -u $(whoami)
