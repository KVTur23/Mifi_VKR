#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --job-name=ft_post
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/code/logs/ft_post_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/code/logs/ft_post_%j.err

set -e

POOL=/mnt/pool/6/kvturanosov/VKR/code
TMP=/tmp/kvt/code

echo "=== $(date) | post: забираю результаты из /tmp в pool ==="

mkdir -p $POOL/results $POOL/Data/finetune_checkpoints $POOL/logs

# Результаты (CSV + preds)
if [ -d "$TMP/results" ]; then
    rsync -a $TMP/results/ $POOL/results/
    echo "results: $(ls $POOL/results | wc -l) файлов"
fi

# Адаптеры (LoRA/QLoRA/AdaLoRA/TinyLoRA) — могут быть большие, ~500MB-2GB каждый
if [ -d "$TMP/code/Data/finetune_checkpoints" ] || [ -d "$TMP/Data/finetune_checkpoints" ]; then
    SRC=$TMP/Data/finetune_checkpoints
    [ -d "$TMP/code/Data/finetune_checkpoints" ] && SRC=$TMP/code/Data/finetune_checkpoints
    rsync -a $SRC/ $POOL/Data/finetune_checkpoints/
    echo "checkpoints: $(du -sh $POOL/Data/finetune_checkpoints | cut -f1)"
fi

echo "--- итог: $POOL/results/finetune_results.csv ---"
[ -f "$POOL/results/finetune_results.csv" ] && cat $POOL/results/finetune_results.csv || echo "(пусто)"
