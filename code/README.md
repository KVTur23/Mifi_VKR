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

Top 10 resuilts

```bash

1) ruadapt_qwen3_32b_qlora_iter3
2) qwen3_14b_lora_iter3
3) qwen3_14b_qlora_iter3
4) qlora_t_pro_it_2_1
5) vikhr_nemo_12b_qlora_iter1
6) vikhr_nemo_12b_lora_iter3
7) vistral_24b_qlora_iter1
8) vistral_24b_adalora_iter1
9) yandexgpt_5_lite_8b_qlora_iter1
10) qwen3_32b_qlora_best

python scripts/run_finetune.py  --config config_models/finetune_configs/ruadapt_qwen3_32b_qlora_iter3

python scripts/run_finetune.py  --config config_models/finetune_configs/qwen3_14b_lora_iter3

python scripts/run_finetune.py  --config config_models/finetune_configs/qwen3_14b_qlora_iter3

python scripts/run_finetune.py  --config config_models/finetune_configs/qlora_t_pro_it_2_1

python scripts/run_finetune.py  --config config_models/finetune_configs/vikhr_nemo_12b_qlora_iter1

python scripts/run_finetune.py  --config config_models/finetune_configs/vikhr_nemo_12b_lora_iter3

python scripts/run_finetune.py  --config config_models/finetune_configs/vistral_24b_qlora_iter1

python scripts/run_finetune.py  --config config_models/finetune_configs/vistral_24b_adalora_iter1

python scripts/run_finetune.py  --config config_models/finetune_configs/yandexgpt_5_lite_8b_qlora_iter1

python scripts/run_finetune.py  --config config_models/finetune_configs/qwen3_32b_qlora_best
```



Результаты сохраняются в:

- адаптер и чекпоинты: `Data/finetune_checkpoints/<run_key>/`;
- метрики: `results/finetune/<run_key>.json`;
- предсказания: `results/preds_<run_key>.csv`.




