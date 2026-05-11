# Ветка для теста на реальных данных
## 1. Установка окружения

```bash
pip install -r requirements.txt
```


## 2. Подготовка данных перед аугментацией

Для чистого запуска аугментации в `Data/` положить `data.json`. 



## 3. Запуск аугментации

Основной запуск:

```bash
python scripts/augmentation.py 
```

Что делает скрипт:

- очищает и подготавливает `Data/data_after_eda.csv`;
- создаёт `Data/train_after_eda.csv` и `Data/data_test.csv`;
- выполняет stage 1, stage 2 и stage 3;
- сохраняет `Data/data_after_stage1.csv`, `Data/data_after_stage2.csv`, `Data/data_after_stage3.csv`;
- считает baseline/augmented метрики классических моделей;
- сохраняет итог в `results/classification_results.csv`.

Если запуск прервался, можно запустить ту же команду ещё раз. Скрипт продолжит с уже созданных файлов в `Data/`.

## 4. Запуск fine-tune

Fine-tune использует:

- train: `Data/data_after_stage3.csv`;
- test: `Data/data_test.csv`;
- конфиг эксперимента из `config_models/finetune_configs/`.

Пока запустить два эксперимента.

```bash
python scripts/run_finetune.py  --config config_models/finetune_configs/qwen3_32b_qlora_no_cw.json

python scripts/run_finetune.py  --config config_models/finetune_configs/qwen3_32b_qlora_cw.json

```



Результаты сохраняются в:

- адаптер и чекпоинты: `Data/finetune_checkpoints/<run_key>/`;
- метрики: `results/finetune/<run_key>.json`;
- предсказания: `results/preds_<run_key>.csv`.




