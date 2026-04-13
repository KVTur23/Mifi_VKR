"""
Zero / One / Few-shot prompt classification.
Дублирует zer0_one_few_shot.ipynb для запуска на сервере через CLI.

Запуск:
    python scripts/run_few_shot.py                        # все модели, все эксперименты
    python scripts/run_few_shot.py --models qwen_14b      # одна модель
    python scripts/run_few_shot.py --models qwen_14b --experiments 0 1 3  # конкретные

Эксперименты (ключ -> описание):
    0       zero-shot
    1       one-shot
    3       few-shot K=3 с описаниями
    5       few-shot K=5 с описаниями (может не влезть)
    "3nd"   few-shot K=3 без описаний
    "5s"    few-shot K=5, примеры 300 символов, с описаниями
    "5snd"  few-shot K=5, примеры 300 символов, без описаний
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Корень проекта — папка выше scripts/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_dataset, load_test_set, TEXT_COL, LABEL_COL
from src.classification.few_shot_examples import (
    prepare_few_shot_examples,
    load_few_shot_examples,
)
from src.classification.prompt_classifier import (
    load_prompt_config,
    load_model,
    unload_model,
    build_prompt,
    classify_dataset,
    load_class_descriptions,
    apply_chat_template,
)
from src.classification.evaluate import evaluate_prompt_classification

DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Конфигурация экспериментов
# Каждый эксперимент: (k, mode, no_desc, max_example_chars)
# ---------------------------------------------------------------------------
EXPERIMENT_CONFIGS = {
    0:      (0, "zero_shot",  False, 500),
    1:      (1, "one_shot",   False, 500),
    3:      (3, "few_shot",   False, 500),
    5:      (5, "few_shot",   False, 500),
    "3nd":  (3, "few_shot",   True,  500),   # K=3 без описаний
    "5s":   (5, "few_shot",   False, 300),   # K=5 короткие примеры с описаниями
    "5snd": (5, "few_shot",   True,  300),   # K=5 короткие примеры без описаний
}

# ---------------------------------------------------------------------------
# Матрица экспериментов: модель -> список ключей из EXPERIMENT_CONFIGS
# ---------------------------------------------------------------------------
DEFAULT_EXPERIMENT_MATRIX = {
    "saiga_8b":     [0, 1],
    "t_lite_8b":    [0, 1, 3, "3nd", "5s", "5snd"],
    "vikhr_12b":    [0, 1, 3, "3nd", "5s", "5snd"],
    "qwen_14b":     [0, 1, 3, "3nd", "5s", "5snd"],
    "qwen_32b":     [0, 1, 3, "3nd", "5s", "5snd"],
}


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def log_raw_responses(df_preds: pd.DataFrame, model_key: str, exp_key, n_samples: int = 10):
    df = df_preds[~df_preds["skipped"]].copy()
    unknowns = df[df["predicted_label"] == "unknown"]
    correct = df[df["predicted_label"] == df["true_label"]]
    wrong = df[
        (df["predicted_label"] != "unknown") &
        (df["predicted_label"] != df["true_label"])
    ]

    print(f"\n{'─'*60}")
    print(f"RAW RESPONSE LOG: {model_key} exp={exp_key}")
    print(f"Correct: {len(correct)} | Wrong: {len(wrong)} | Unknown: {len(unknowns)}")
    print(f"{'─'*60}")

    if len(unknowns) > 0:
        print(f"\n  UNKNOWN ({len(unknowns)} шт, первые {min(n_samples, len(unknowns))}):")
        for _, row in unknowns.head(n_samples).iterrows():
            raw = str(row["raw_response"])[:150]
            print(f"    true: {row['true_label']}")
            print(f"    raw:  {raw}")
            print()

    if len(wrong) > 0:
        print(f"  WRONG ({len(wrong)} шт, первые {min(n_samples, len(wrong))}):")
        for _, row in wrong.head(n_samples).iterrows():
            raw = str(row["raw_response"])[:150]
            print(f"    true: {row['true_label']}")
            print(f"    pred: {row['predicted_label']}")
            print(f"    raw:  {raw}")
            print()

    if len(correct) > 0:
        print(f"  CORRECT (первые {min(3, len(correct))}):")
        for _, row in correct.head(3).iterrows():
            raw = str(row["raw_response"])[:150]
            print(f"    true: {row['true_label']}")
            print(f"    raw:  {raw}")
            print()

    print(f"{'─'*60}")


def run_experiment(
    model_key: str,
    exp_key,
    model,
    tokenizer,
    model_cfg: dict,
    cfg: dict,
    df_test: pd.DataFrame,
    labels: list,
    descriptions: dict,
    fs_data: dict,
    groups: dict,
) -> dict:
    max_context = model_cfg["max_context"]
    gen_params = cfg["generation_params"]

    k_shot, mode, no_desc, max_example_chars = EXPERIMENT_CONFIGS[exp_key]

    if k_shot == 0:
        examples = None
    elif k_shot == 1:
        examples = fs_data["examples"]["1"]
    else:
        examples = fs_data["examples"][str(k_shot)]

    print(f"\n{'='*60}")
    print(f"Эксперимент: {model_key} | exp={exp_key} K={k_shot} mode={mode} no_desc={no_desc} max_chars={max_example_chars}")
    print(f"{'='*60}")

    test_prompt = build_prompt(
        df_test.iloc[0][TEXT_COL], labels, descriptions, mode, examples,
        no_desc=no_desc, max_example_chars=max_example_chars,
    )
    test_prompt_formatted = apply_chat_template(tokenizer, test_prompt)
    test_tokens = len(tokenizer.encode(test_prompt_formatted))
    print(f"Длина тестового промпта: {test_tokens} токенов (лимит: {max_context})")

    if test_tokens >= max_context - 100:
        print(f"SKIP: промпт ({test_tokens}) >= лимит ({max_context - 100})")
        return {
            "model": model_key,
            "model_name": model_cfg["model_name"],
            "model_size": model_cfg["vram_gb"],
            "exp_key": str(exp_key),
            "k_shots": k_shot,
            "no_desc": no_desc,
            "max_example_chars": max_example_chars,
            "skipped": True,
            "prompt_tokens": test_tokens,
            "n_test": len(df_test),
        }

    df_preds = classify_dataset(
        df_test, model, tokenizer, labels, descriptions,
        mode=mode, gen_params=gen_params,
        max_context=max_context, examples=examples,
        fuzzy_cutoff=cfg["extract_prediction"]["fuzzy_cutoff"],
        no_desc=no_desc, max_example_chars=max_example_chars,
    )

    metrics = evaluate_prompt_classification(df_preds, groups)
    log_raw_responses(df_preds, model_key, exp_key)

    preds_path = RESULTS_DIR / f"preds_{model_key}_exp{exp_key}.csv"
    df_preds.to_csv(preds_path, index=False)

    return {
        "model": model_key,
        "model_name": model_cfg["model_name"],
        "model_size": model_cfg["vram_gb"],
        "exp_key": str(exp_key),
        "k_shots": k_shot,
        "no_desc": no_desc,
        "max_example_chars": max_example_chars,
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "unknown_rate": metrics["unknown_rate"],
        "f1_group_A": metrics.get("f1_group_A"),
        "f1_group_B": metrics.get("f1_group_B"),
        "f1_group_C": metrics.get("f1_group_C"),
        "prompt_tokens": test_tokens,
        "skipped": False,
        "n_test": len(df_test),
    }


# ---------------------------------------------------------------------------
# Шаг 1: Описания классов
# ---------------------------------------------------------------------------

def ensure_class_descriptions(labels: list, df_train: pd.DataFrame, cfg: dict) -> dict:
    desc_path = DATA_DIR / "class_descriptions.json"
    if desc_path.exists():
        descriptions = load_class_descriptions(desc_path)
        print(f"Описания загружены из кэша ({len(descriptions)} классов)")
        return descriptions

    from src.augmentation.stage1_llm_generate import generate_class_context
    from src.augmentation.llm_utils import load_llm

    config_path = str(PROJECT_ROOT / "config_models" / "aug_configs" / "model_vllm_32b.json")
    llm, sampling_params, system_prompt = load_llm(config_path)

    descriptions = {}
    for cls in sorted(labels):
        examples = df_train[df_train[LABEL_COL] == cls][TEXT_COL].tolist()[:5]
        desc = generate_class_context(cls, examples, llm, sampling_params, system_prompt)
        descriptions[cls] = desc

    with open(desc_path, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)
    print(f"Сгенерировано и сохранено {len(descriptions)} описаний")

    del llm, sampling_params, system_prompt
    gc.collect()
    torch.cuda.empty_cache()
    return descriptions


# ---------------------------------------------------------------------------
# Шаг 2: Few-shot примеры
# ---------------------------------------------------------------------------

def ensure_few_shot_examples() -> dict:
    fs_path = DATA_DIR / "few_shot_examples.json"
    if fs_path.exists():
        fs_data = load_few_shot_examples(fs_path)
        print("Few-shot примеры загружены из кэша")
    else:
        fs_data = prepare_few_shot_examples(k_values=[1, 3, 5])
        print("Few-shot примеры сгенерированы и сохранены")

    for k in ["1", "3", "5"]:
        total = sum(len(v) for v in fs_data["examples"][k].values())
        print(f"K={k}: {total} примеров")
    return fs_data


# ---------------------------------------------------------------------------
# Шаг 4: Сводная таблица
# ---------------------------------------------------------------------------

def build_comparison_table():
    df_prompt = pd.read_csv(RESULTS_DIR / "prompt_results.csv")
    df_prompt = df_prompt[~df_prompt["skipped"].astype(bool)]

    print("\nРезультаты prompt-based классификации:")
    display_cols = [
        "model", "exp_key", "k_shots", "no_desc", "balanced_accuracy",
        "macro_f1", "unknown_rate", "f1_group_A", "f1_group_B", "f1_group_C",
    ]
    print(df_prompt[display_cols].round(4).to_string(index=False))

    baseline_path = RESULTS_DIR / "classification_results.csv"
    if not baseline_path.exists():
        print("\nBaseline results не найдены — сводная таблица не создаётся.")
        return

    df_baseline = pd.read_csv(baseline_path)
    rows = []
    for _, r in df_baseline.iterrows():
        rows.append({
            "method": "baseline" if r["stage"] == "baseline" else "augmented",
            "model": r["model"],
            "setting": r["stage"],
            "balanced_accuracy": r["balanced_accuracy"],
            "macro_f1": r["macro_f1"],
            "unknown_rate": 0.0,
        })
    for _, r in df_prompt.iterrows():
        rows.append({
            "method": "prompt",
            "model": r["model_name"],
            "setting": f"K={r['k_shots']}_exp{r['exp_key']}",
            "balanced_accuracy": r["balanced_accuracy"],
            "macro_f1": r["macro_f1"],
            "unknown_rate": r["unknown_rate"],
        })

    df_all = pd.DataFrame(rows)
    df_all.to_csv(RESULTS_DIR / "all_methods_comparison.csv", index=False)
    print("\nСводная таблица сохранена в results/all_methods_comparison.csv")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Zero/One/Few-shot prompt classification")
    parser.add_argument(
        "--models", nargs="+",
        choices=list(DEFAULT_EXPERIMENT_MATRIX.keys()),
        default=None,
        help="Модели для запуска (по умолчанию — все)",
    )
    parser.add_argument(
        "--experiments", nargs="+",
        default=None,
        help="Эксперименты для запуска: 0 1 3 5 3nd 5s 5snd",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.models:
        experiment_matrix = {m: DEFAULT_EXPERIMENT_MATRIX[m] for m in args.models}
    else:
        experiment_matrix = dict(DEFAULT_EXPERIMENT_MATRIX)

    if args.experiments:
        # Конвертируем строки в int где возможно
        selected = []
        for e in args.experiments:
            try:
                selected.append(int(e))
            except ValueError:
                selected.append(e)
        experiment_matrix = {
            m: [k for k in ks if k in selected]
            for m, ks in experiment_matrix.items()
        }
        experiment_matrix = {m: ks for m, ks in experiment_matrix.items() if ks}

    # Загрузка конфига и данных
    cfg = load_prompt_config()
    prompt_models = cfg["prompt_models"]

    df_train = load_dataset(stage=0)
    df_test = load_test_set()
    labels = sorted(df_train[LABEL_COL].unique().tolist())

    print(f"Тест: {len(df_test)} примеров, Классов: {len(labels)}")

    descriptions = ensure_class_descriptions(labels, df_train, cfg)
    fs_data = ensure_few_shot_examples()
    groups = fs_data["groups"]

    groups_counts = {g: sum(1 for v in groups.values() if v == g) for g in "ABC"}
    print(f"Группы: A={groups_counts['A']}, B={groups_counts['B']}, C={groups_counts['C']}")

    # Загружаем уже посчитанные
    results_path = RESULTS_DIR / "prompt_results.csv"
    if results_path.exists():
        df_existing = pd.read_csv(results_path)
        # Совместимость со старым форматом где не было exp_key
        if "exp_key" not in df_existing.columns:
            df_existing["exp_key"] = df_existing["k_shots"].astype(str)
        all_results = df_existing.to_dict("records")
        done = {(r["model"], str(r["exp_key"])) for r in all_results}
        print(f"Загружено {len(all_results)} готовых экспериментов: {done}")
    else:
        all_results = []
        done = set()
        print("Предыдущих результатов нет, запуск с нуля")

    for model_key, exp_keys in experiment_matrix.items():
        model_cfg = prompt_models[model_key]
        exp_todo = [e for e in exp_keys if (model_key, str(e)) not in done]

        if not exp_todo:
            print(f"\n[SKIP] {model_cfg['model_name']} — все эксперименты уже посчитаны")
            continue

        print(f"\n{'#'*60}")
        print(f"# МОДЕЛЬ: {model_cfg['model_name']}")
        print(f"# Эксперименты: {exp_todo}")
        print(f"{'#'*60}")

        try:
            model, tokenizer = load_model(model_cfg)
        except Exception as e:
            print(f"[ERROR] Не удалось загрузить {model_cfg['model_name']}: {e}")
            print("Пропускаю, перехожу к следующей модели...")
            continue

        for exp_key in exp_todo:
            result = run_experiment(
                model_key, exp_key, model, tokenizer, model_cfg,
                cfg, df_test, labels, descriptions, fs_data, groups,
            )
            all_results.append(result)
            done.add((model_key, str(exp_key)))
            pd.DataFrame(all_results).to_csv(results_path, index=False)

        unload_model(model, tokenizer)

    print(f"\nВсего экспериментов: {len(all_results)}")
    build_comparison_table()


if __name__ == "__main__":
    main()
