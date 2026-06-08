#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --job-name=ft_pre
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/code/logs/ft_pre_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/code/logs/ft_pre_%j.err

set -e

# Локальная рабочая область на SSD ноды — копируем сюда модели и код,
# чтобы comp-джобы читали с локального диска, а не по NFS.
mkdir -p /tmp/kvt/models /tmp/kvt/hf /tmp/kvt/cache

echo "=== $(date) | pre: rsync моделей в /tmp ==="
rsync -a --info=progress2 /mnt/pool/6/kvturanosov/VKR/models/ /tmp/kvt/models/
echo "Модели: $(du -sh /tmp/kvt/models | cut -f1)"
ls /tmp/kvt/models | head -20

echo "=== $(date) | pre: копирую код в /tmp ==="
rm -rf /tmp/kvt/code
cp -r /mnt/pool/6/kvturanosov/VKR/code /tmp/kvt/code
# Результаты/чекпоинты будем писать в /tmp, потом post вернёт в pool
mkdir -p /tmp/kvt/code/results /tmp/kvt/code/Data/finetune_checkpoints /tmp/kvt/code/logs
echo "Готово: /tmp/kvt/code"
