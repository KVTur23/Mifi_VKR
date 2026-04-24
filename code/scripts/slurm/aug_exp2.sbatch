#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=aug_exp2
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/code/logs/aug_exp2_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/code/logs/aug_exp2_%j.err

# exp-aug-2: RuadaptQwen3-32B-Instruct-AWQ на всех этапах LLM + NLLB-3.3B на переводе
# Pre-синхронизация — отдельным джобом aug_pre.sbatch (ветка exp-aug-1).

set -e

POOL=/mnt/pool/6/kvturanosov
CODE_DST=/tmp/kvt/exp2/code
MODELS_DST=/tmp/kvt/models
EXP_CFG=config_models/aug_configs/model_aug_exp2.json

export HOME=$POOL
export XDG_CACHE_HOME=/tmp/kvt/exp2/cache
export HF_HOME=/tmp/kvt/exp2/hf
export HF_HUB_CACHE=$MODELS_DST
export HUGGINGFACE_HUB_CACHE=$MODELS_DST
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "=== SLURM $SLURM_JOB_ID | host=$(hostname) | $(date) ==="
nvidia-smi | head -20

source $POOL/VKR/venv/bin/activate
cd $CODE_DST

echo "=== $(date) | STAGE 1: LLM-генерация ==="
python -u src/augmentation/stage1_llm_generate.py --config $EXP_CFG

echo "=== $(date) | STAGE 2: парафраз ==="
python -u src/augmentation/stage2_paraphrase.py --config $EXP_CFG

echo "=== $(date) | STAGE 3: обратный перевод ==="
python -u src/augmentation/stage3_back_translation.py --config $EXP_CFG

echo "=== $(date) | POST: rsync результатов в pool ==="
RESULTS_DST=$POOL/VKR/results/aug_exp2
DATA_DST=$POOL/VKR/data_aug/exp2
mkdir -p $RESULTS_DST $DATA_DST
[ -d "$CODE_DST/Data" ] && rsync -a $CODE_DST/Data/ $DATA_DST/ && echo "data: $(du -sh $DATA_DST | cut -f1)"
[ -d "$CODE_DST/results" ] && rsync -a $CODE_DST/results/ $RESULTS_DST/ && echo "results: $(ls $RESULTS_DST | wc -l) файлов"

echo "=== $(date) | aug_exp2 DONE ==="
