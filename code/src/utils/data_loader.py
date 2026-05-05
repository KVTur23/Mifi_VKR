"""
data_loader.py — Загрузка данных и работа с чекпоинтами

Центральный модуль для загрузки датасета на любом этапе пайплайна.
Каждый этап аугментации сохраняет промежуточный CSV (data_after_stageN.csv).
Этот модуль умеет подхватывать нужный чекпоинт, чтобы не прогонять
уже пройденные этапы заново.

Также здесь утилиты для анализа распределения классов, 
чтобы понять, каким классам ещё требуется аугментация.
"""

import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path


# --- Константы ---

TEXT_COL = "text"       # Колонка с текстом письма
LABEL_COL = "label"     # Колонка с меткой класса
RANDOM_SEED = 42        # Единый seed для воспроизводимости по всему проекту

TEST_FILE = "data_test.csv"          # Тестовая выборка (не аугментируется)
ORIGINAL_FILE = "data_after_eda.csv" # Оригинальный датасет (не трогаем)

# Маппинг этапов на файлы — порядок важен: от самого свежего к исходному
STAGE_FILES = {
    3: "data_after_stage3.csv",
    2: "data_after_stage2.csv",
    1: "data_after_stage1.csv",
    0: "train_after_eda.csv",      # Train-часть после разбиения
}

# Путь к папке с данными — Data/ лежит рядом с src/
DATA_DIR = Path(__file__).parent.parent.parent / "Data"


def load_dataset(stage: int, data_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Загружает датасет для указанного этапа с учётом чекпоинтов.

    Логика такая: если для этапа N уже есть файл data_after_stageN.csv —
    берём его (значит этап уже пройден или частично пройден). Если нет —
    откатываемся к предыдущему этапу, и так до исходного файла.

    Например, если запрашиваем stage=2, а файла data_after_stage2.csv нет,
    но есть data_after_stage1.csv — загрузим его.

    Аргументы:
        stage:    номер этапа (1, 2 или 3). Для загрузки исходного — 0.
        data_dir: путь к папке Data/. Если None, используется дефолтный
                  путь относительно расположения этого файла.

    Возвращает:
        DataFrame с колонками text и label (как минимум)
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    if stage not in STAGE_FILES:
        raise ValueError(
            f"Неизвестный этап: {stage}. Допустимые значения: {list(STAGE_FILES.keys())}"
        )

    # Идём от запрошенного этапа вниз, ищем первый существующий файл
    for s in range(stage, -1, -1):
        file_path = data_dir / STAGE_FILES[s]
        if file_path.exists():
            df = pd.read_csv(file_path)
            _validate_columns(df, file_path)

            if s == stage:
                print(f"[Данные] Найден чекпоинт этапа {stage}: {file_path.name} "
                      f"({len(df)} записей)")
            else:
                print(f"[Данные] Чекпоинт этапа {stage} не найден, "
                      f"загружен этап {s}: {file_path.name} "
                      f"({len(df)} записей)")
            return df

    # если вообще нет ни одного файла
    raise FileNotFoundError(
        f"Не найден ни один файл данных в {data_dir}. "
        f"Проверь, что data_after_eda.csv или train_after_eda.csv на месте"
    )


def save_checkpoint(df: pd.DataFrame, stage: int, data_dir: str | Path | None = None) -> Path:
    """
    Сохраняет датасет как чекпоинт после завершения или во время этапа,
    на случай падения ядра.

    Аргументы:
        df:       DataFrame для сохранения
        stage:    номер этапа (1, 2 или 3)
        data_dir: путь к папке Data/

    Возвращает:
        Path до сохранённого файла
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    if stage not in STAGE_FILES or stage == 0:
        raise ValueError(f"Сохранять можно только этапы 1–3, получен: {stage}")

    file_path = data_dir / STAGE_FILES[stage]
    df.to_csv(file_path, index=False)
    print(f"[Данные] Сохранён чекпоинт этапа {stage}: {file_path.name} ({len(df)} записей)")
    mirror_file_to_aug_pool(file_path)
    return file_path


def mirror_file_to_aug_pool(file_path: str | Path, prefix: str = "[Данные]") -> Path | None:
    """
    Copies a live augmentation artifact from /tmp/kvt back to the pool run Data dir.

    SLURM jobs keep the active workspace in fast tmp storage. If vLLM/NLLB fails
    before post.sbatch runs, the latest cache/checkpoint would otherwise be lost.
    The pool target is enabled only when AUG_POOL_DATA_DIR is set by sbatch.
    """
    pool_data_dir = os.environ.get("AUG_POOL_DATA_DIR")
    if not pool_data_dir:
        return None

    src = Path(file_path)
    if not src.exists():
        return None

    dst_dir = Path(pool_data_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    try:
        if src.resolve() == dst.resolve():
            return dst
    except OSError:
        pass

    tmp = dst.with_name(f"{dst.name}.tmp")
    try:
        shutil.copy2(src, tmp)
        tmp.replace(dst)
        print(f"{prefix} Зеркальная копия в pool: {dst}")
        return dst
    except Exception as e:
        print(f"{prefix} Не удалось скопировать {src.name} в pool: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def remove_aug_pool_file(filename: str, prefix: str = "[Данные]") -> None:
    """Removes a stale live artifact from AUG_POOL_DATA_DIR if pool mirroring is enabled."""
    pool_data_dir = os.environ.get("AUG_POOL_DATA_DIR")
    if not pool_data_dir:
        return

    path = Path(pool_data_dir) / filename
    try:
        if path.exists():
            path.unlink()
            print(f"{prefix} Удалён live-файл из pool: {path}")
    except Exception as e:
        print(f"{prefix} Не удалось удалить {path}: {e}")


def get_class_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Считает, сколько примеров в каждом классе.

    Аргументы:
        df: DataFrame с колонкой label

    Возвращает:
        pd.Series с количеством примеров по классам (от большего к меньшему)
    """
    return df[LABEL_COL].value_counts().sort_values(ascending=False)


def get_classes_to_augment(
    df: pd.DataFrame,
    min_count: int,
    max_count: int
) -> dict[str, int]:
    """
    Находит классы, которым нужна аугментация в заданном диапазоне.

    Возвращает словарь: ключ — имя класса, значение — сколько примеров
    в нём сейчас. В словарь попадают только классы, где количество
    примеров >= min_count и < max_count.

    - Этап 1: min_count=0, max_count=15
    - Этап 2: min_count=0, max_count=35  (после этапа 1 все >= 15)
    - Этап 3: min_count=0, max_count=50  (после этапа 2 все >= 35)

    Аргументы:
        df:        DataFrame с колонкой label
        min_count: нижняя граница количества примеров (включительно)
        max_count: верхняя граница (не включительно) — классы с >= max_count
                   уже не нуждаются в аугментации

    Возвращает:
        Словарь {имя_класса: текущее_количество_примеров}
    """
    distribution = get_class_distribution(df)

    classes = {}
    for class_name, count in distribution.items():
        if min_count <= count < max_count:
            classes[class_name] = count

    return classes


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Стратифицированное разбиение на train/test с гарантией минимум
    1 примера на класс в каждой части.

    Читает переданный df, сохраняет:
      - train → train_after_eda.csv (STAGE_FILES[0])
      - test  → data_test.csv
    Оригинал data_after_eda.csv остаётся нетронутым.

    Возвращает (train_df, test_df).
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    # Разбиение по классам
    rng = np.random.RandomState(RANDOM_SEED)
    train_idx, test_idx = [], []

    for cls in sorted(df[LABEL_COL].unique()):
        cls_indices = df[df[LABEL_COL] == cls].index.tolist()
        rng.shuffle(cls_indices)
        n_test = max(1, int(len(cls_indices) * test_size))
        test_idx.extend(cls_indices[:n_test])
        train_idx.extend(cls_indices[n_test:])

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    # Сохраняем train и test
    train_df.to_csv(data_dir / STAGE_FILES[0], index=False)
    test_df.to_csv(data_dir / TEST_FILE, index=False)

    print(f"[Данные] Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df


def load_test_set(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Загружает тестовую выборку из data_test.csv."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    path = data_dir / TEST_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Тестовая выборка не найдена: {path}. "
            f"Сначала выполни split_train_test()."
        )

    df = pd.read_csv(path)
    _validate_columns(df, path)
    print(f"[Данные] Тестовая выборка: {path.name} ({len(df)} записей)")
    return df


def _validate_columns(df: pd.DataFrame, file_path: Path) -> None:
    """
    Проверяет, что в DataFrame есть обязательные колонки text и label.
    """
    missing = []
    if TEXT_COL not in df.columns:
        missing.append(TEXT_COL)
    if LABEL_COL not in df.columns:
        missing.append(LABEL_COL)

    if missing:
        raise KeyError(
            f"В файле {file_path} не найдены колонки: {', '.join(missing)}. "
            f"Ожидаются колонки '{TEXT_COL}' и '{LABEL_COL}'"
        )
