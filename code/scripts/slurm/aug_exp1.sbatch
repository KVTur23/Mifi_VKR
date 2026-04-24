#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=aug_exp1
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/code/logs/aug_exp1_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/code/logs/aug_exp1_%j.err

# exp-aug-1: Qwen3-32B-AWQ на всех этапах (stage1/stage2/stage3-судья) + NLLB-3.3B на переводе

set -e

POOL=/mnt/pool/6/kvturanosov
CODE_SRC=$POOL/VKR/worktrees/exp-aug-1/code
MODELS_SRC=$POOL/VKR/models
CODE_DST=/tmp/kvt/exp1/code
MODELS_DST=/tmp/kvt/models
EXP_CFG=config_models/aug_configs/model_aug_exp1.json

# Нужные модели для этого эксперимента — rsync только их
NEEDED_MODELS=(
  "models--Qwen--Qwen3-32B-AWQ"
  "models--facebook--nllb-200-3.3B"
  "models--ai-forever--sbert_large_nlu_ru"
)

# per-exp окружение, чтобы 4 параллельных джоба не пересекались в cache
export HOME=$POOL
export XDG_CACHE_HOME=/tmp/kvt/exp1/cache
export HF_HOME=/tmp/kvt/exp1/hf
export HF_HUB_CACHE=$MODELS_DST
export HUGGINGFACE_HUB_CACHE=$MODELS_DST
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "=== SLURM $SLURM_JOB_ID | host=$(hostname) | $(date) ==="
nvidia-smi | head -20

# ============================================================
# PRE: rsync кода и моделей из pool в /tmp (локальный SSD ноды)
# ============================================================
echo "=== $(date) | PRE: rsync кода и моделей ==="
mkdir -p $CODE_DST $MODELS_DST $XDG_CACHE_HOME $HF_HOME

rm -rf $CODE_DST
cp -r $CODE_SRC $CODE_DST
mkdir -p $CODE_DST/Data $CODE_DST/logs $CODE_DST/results

# Shared /tmp/kvt/models между 4 параллельными джобами.
# flock сериализует rsync — первый джоб копирует, остальные ждут, потом видят
# готовые файлы и быстро проскакивают. Избегаем race-condition при write.
(
  flock 200
  echo "  [flock] got models lock"
  for m in "${NEEDED_MODELS[@]}"; do
    if [ -d "$MODELS_SRC/$m" ]; then
      rsync -a "$MODELS_SRC/$m/" "$MODELS_DST/$m/"
    else
      echo "  [WARN] $MODELS_SRC/$m не найдена — пропускаю"
    fi
  done
  echo "  [flock] models sync done"
) 200>/tmp/kvt/models.lock
echo "Модели в /tmp: $(du -sh $MODELS_DST | cut -f1)"

source $POOL/VKR/venv/bin/activate
cd $CODE_DST

# ============================================================
# STAGES 1 → 2 → 3
# ============================================================
echo "=== $(date) | STAGE 1: LLM-генерация ==="
python -u src/augmentation/stage1_llm_generate.py --config $EXP_CFG

echo "=== $(date) | STAGE 2: парафраз ==="
python -u src/augmentation/stage2_paraphrase.py --config $EXP_CFG

echo "=== $(date) | STAGE 3: обратный перевод ==="
python -u src/augmentation/stage3_back_translation.py --config $EXP_CFG

# ============================================================
# POST: забираем результаты обратно в pool
# ============================================================
echo "=== $(date) | POST: rsync результатов в pool ==="
RESULTS_DST=$POOL/VKR/results/aug_exp1
DATA_DST=$POOL/VKR/data_aug/exp1

mkdir -p $RESULTS_DST $DATA_DST

[ -d "$CODE_DST/Data" ] && rsync -a $CODE_DST/Data/ $DATA_DST/ && echo "data: $(du -sh $DATA_DST | cut -f1)"
[ -d "$CODE_DST/results" ] && rsync -a $CODE_DST/results/ $RESULTS_DST/ && echo "results: $(ls $RESULTS_DST | wc -l) файлов"

echo "=== $(date) | aug_exp1 DONE ==="
