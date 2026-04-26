#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=aug_exp1
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/code/logs/aug_exp1_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/code/logs/aug_exp1_%j.err

# exp-aug-1: Qwen3-32B-AWQ на всех этапах (stage1/stage2/stage3-судья) + NLLB distilled-1.3B на переводе
# Pre-синхронизация кода и моделей делается отдельным джобом aug_pre.sbatch (см. эту же папку).

set -e

POOL=/mnt/pool/6/kvturanosov
CODE_DST=/tmp/kvt/exp1/code
MODELS_DST=/tmp/kvt/models
EXP_CFG=config_models/aug_configs/model_aug_exp1.json

# per-exp окружение, чтобы 4 параллельных джоба не пересекались в cache
export HOME=/tmp/kvt/exp1/home
export TRITON_CACHE_DIR=/tmp/kvt/exp1/triton
export XDG_CACHE_HOME=/tmp/kvt/exp1/cache
export HF_HOME=/tmp/kvt/exp1/hf
export HF_HUB_CACHE=$MODELS_DST
export HUGGINGFACE_HUB_CACHE=$MODELS_DST
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

mkdir -p "$HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME"

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
RESULTS_DST=$POOL/VKR/results/aug_exp1
DATA_DST=$POOL/VKR/data_aug/exp1
mkdir -p $RESULTS_DST $DATA_DST
[ -d "$CODE_DST/Data" ] && rsync -a $CODE_DST/Data/ $DATA_DST/ && echo "data: $(du -sh $DATA_DST | cut -f1)"
[ -d "$CODE_DST/results" ] && rsync -a $CODE_DST/results/ $RESULTS_DST/ && echo "results: $(ls $RESULTS_DST | wc -l) файлов"

echo "=== $(date) | aug_exp1 DONE ==="
