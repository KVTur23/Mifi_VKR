"""
data_loader.py — Загрузка данных и работа с чекпоинтами

Центральный модуль для загрузки датасета на любом этапе пайплайна.
Каждый этап аугментации сохраняет промежуточный CSV (data_after_stageN.csv).
Этот модуль умеет подхватывать нужный чекпоинт, чтобы не прогонять
уже пройденные этапы заново.

Также здесь живут утилиты для анализа распределения классов —
они нужны практически в каждом этапе, чтобы понять, каким классам
ещё требуется аугментация.
"""

import pandas as pd
from pathlib import Path


# --- Константы ---

TEXT_COL = "text"       # Колонка с текстом письма
LABEL_COL = "label"     # Колонка с меткой класса
RANDOM_SEED = 42        # Единый seed для воспроизводимости по всему проекту

# Маппинг этапов на файлы — порядок важен: от самого свежего к исходному
STAGE_FILES = {
    3: "data_after_stage3.csv",
    2: "data_after_stage2.csv",
    1: "data_after_stage1.csv",
    0: "data_after_eda.csv",       # Исходный датасет после EDA
}

# Путь к папке с данными (относительно корня проекта)
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

    # Сюда попадём, только если вообще нет ни одного файла
    raise FileNotFoundError(
        f"Не найден ни один файл данных в {data_dir}. "
        f"Проверь, что data_after_eda.csv на месте"
    )


def save_checkpoint(df: pd.DataFrame, stage: int, data_dir: str | Path | None = None) -> Path:
    """
    Сохраняет датасет как чекпоинт после завершения этапа.

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
    return file_path


def get_class_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Считает, сколько примеров в каждом классе.

    Возвращает Series: индекс — название класса, значение — количество.
    Отсортировано по убыванию, чтобы сразу видеть, где густо, а где пусто.

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

    Это нужно, чтобы каждый этап работал только со «своими» классами:
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


def _validate_columns(df: pd.DataFrame, file_path: Path) -> None:
    """
    Проверяет, что в DataFrame есть обязательные колонки text и label.

    Если колонок нет — падаем с понятной ошибкой, а не с загадочным KeyError
    где-нибудь в середине пайплайна.
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
