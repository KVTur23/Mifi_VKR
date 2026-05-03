ТЗ для Codex: Реаугментация данных Stage 2 + Stage 3
0. Контекст проекта
Это дипломная работа (ВКР) по классификации корпоративной почты на 36 классов. Текущий fine-tune (QLoRA Qwen3-32B + focal loss + class_weights) даёт macro_f1=0.541 на test. Причина просадки — некачественная аугментация в Stage 2 и Stage 3. Цель этой задачи: переделать Stage 2 и Stage 3 на качественные LLM-парафразы и пересобрать датасет.

Рабочая директория: /Users/kvt/Documents/VKR/code/

1. Текущее состояние данных (НЕ менять оригиналы)
Файлы в Data/:

Файл	Строк	Описание
train_after_eda.csv	1409	Оригинальные письма (НЕ ТРОГАТЬ)
data_test.csv	341	Test set (НЕ ТРОГАТЬ)
data_after_stage1.csv	1533	После Stage 1 (LLM-genеration для group C). Используется как вход для нового Stage 2 v2
data_after_stage2.csv	1941	СТАРЫЙ Stage 2 — будет заменён
data_after_stage3.csv	2331	СТАРЫЙ Stage 3 (NLLB, сломан) — будет заменён
Группы классов (по train_after_eda.csv):

Group A (≥50 orig): 9 классов, 985 примеров — без аугментации
Group B (15-49 orig): 15 классов, 368 примеров — нуждается в Stage 2 + Stage 3
Group C (<15 orig): 12 классов, 56 примеров — Stage 1 + Stage 2 + Stage 3
Найденные проблемы:

В data_after_stage3.csv: 224 строки (9.6%) содержат сломанные плейсхолдеры от NLLB: <10>, <FD, NEFT (вместо НЕФТЬ), OKPO (вместо ОКПО), и т.д.
В data_after_stage2.csv: 5.5% текстов обрезаны на середине слова
Медианная длина синтетики 1027 симв vs test 1288 (-17%)
p95 длины синтетики 2501 vs test 3797 (-38%)
2. Цель и итоговый артефакт
Создать Data/data_after_stage3_v2.csv — заменитель data_after_stage3.csv, в котором:

0 broken placeholders
Все классы B и C на ≥50 примерах (target 50, до 60 ОК)
Group A не трогаем (остаётся как есть)
Распределение длин close to test (медиана 1100-1500, p95 ≥3200)
Затем запустить FT и сравнить с baseline (focal best 0.541).

3. Архитектура нового pipeline

data_after_stage1.csv (1533 строк, не меняется)
        ↓
[НОВЫЙ Stage 2 v2: light paraphrase через Qwen2.5-32B-AWQ]
        ↓ + 408 чистых парафразов для B/C до 35/класс
data_after_stage2_v2.csv (~1941 строк)
        ↓
[НОВЫЙ Stage 3 v2: deep paraphrase через тот же Qwen2.5-32B-AWQ, ИНАЧЕ ПРОМПТ]
        ↓ + 405 чистых deep-парафразов для B/C до 50/класс
data_after_stage3_v2.csv (~2330 строк, чистых)
Ключевое отличие Stage 2 v2 от Stage 3 v2: оба используют LLM-парафраз, но Stage 2 v2 — лёгкий (синоним-замены, минимальная структурная правка, sim 0.85-0.95), Stage 3 v2 — глубокий (другая структура, перестановка абзацев, ролевая перспектива, sim 0.70-0.85). Это даёт естественное разнообразие в датасете.

NLLB вообще не используется. stage3_back_translation.py остаётся в репо для воспроизводимости старых ранов, но не вызывается.

ШАГ 1 — Расширить validation.py (3 новых фильтра)
Файл: src/augmentation/validation.py

1.1 Добавить новый фильтр broken placeholders
В начало файла, рядом с другими константами:


# Сломанные плейсхолдеры от NLLB и подобного
_BROKEN_PLACEHOLDERS_RE = re.compile(
    r'<\d+>?'                  # <10>, <9, <15
    r'|<[A-Z]{2,5}\b'          # <FD, <NUM, <ORG (без закрывающей >)
    r'|\bNEFT\b'               # транслитерация НЕФТЬ
    r'|\bGAZ\b'                # транслитерация ГАЗ
    r'|\bOKPO\b'               # транслитерация ОКПО
    r'|\bOGRN\b'               # транслитерация ОГРН
    r'|\bINN\b'                # транслитерация ИНН
    r'|\bKPP\b'                # транслитерация КПП
    r'|\bBIK\b'                # транслитерация БИК
)
Добавить функцию (после filter_foreign_scripts):


def filter_broken_placeholders(
    texts: list[str],
    class_name: str,
) -> list[str]:
    """Фильтр 8: Сломанные плейсхолдеры (broken placeholders) от back-translation.
    
    Удаляет тексты, содержащие <10>, <FD, NEFT, OKPO и подобные артефакты
    транслитерации/токенизации, которые появляются после NLLB обратного перевода.
    """
    filtered = [t for t in texts if not _BROKEN_PLACEHOLDERS_RE.search(t)]
    removed = len(texts) - len(filtered)
    if removed > 0:
        print(f"[Валидация] Класс «{class_name}»: broken_placeholders отсеяно {removed}")
    return filtered
1.2 Добавить фильтр максимальной длины

MAX_TEXT_LENGTH = 5000  # выше этого — вероятно галлюцинация LLM

def filter_max_length(
    texts: list[str],
    class_name: str,
    max_length: int = MAX_TEXT_LENGTH,
) -> list[str]:
    """Фильтр 9: Слишком длинные тексты (likely галлюцинация LLM)."""
    filtered = [t for t in texts if len(t.strip()) <= max_length]
    removed = len(texts) - len(filtered)
    if removed > 0:
        print(f"[Валидация] Класс «{class_name}»: max_length отсеяно {removed}")
    return filtered
1.3 Добавить фильтр обрезанных на середине слова

_TRUNCATED_END_RE = re.compile(
    r'[а-яa-z]{3,}\s*$',  # текст заканчивается на середине слова в нижнем регистре
    re.IGNORECASE,
)

def filter_truncated(
    texts: list[str],
    class_name: str,
) -> list[str]:
    """Фильтр 10: Текст обрезан на середине слова (truncated mid-word).
    
    Письмо должно заканчиваться знаком препинания (. ! ?) или специальной 
    конструкцией (подпись, [PERSON]). Если заканчивается на середине слова — 
    LLM не дописал, отбрасываем.
    """
    def is_proper_ending(text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        # Допустимые окончания
        if text[-1] in '.!?»"':
            return True
        # Заканчивается плейсхолдером
        if text.endswith(']') and '[' in text[-30:]:
            return True
        return False
    
    filtered = [t for t in texts if is_proper_ending(t)]
    removed = len(texts) - len(filtered)
    if removed > 0:
        print(f"[Валидация] Класс «{class_name}»: truncated отсеяно {removed}")
    return filtered
1.4 Обновить validate_generated_texts() — добавить вызовы новых фильтров
В функции validate_generated_texts, в секции "Цепочка фильтров", после filter_foreign_scripts (строка 86) добавить:


texts = filter_foreign_scripts(texts, class_name)
texts = filter_broken_placeholders(texts, class_name)   # НОВОЕ — фильтр 8
texts = filter_max_length(texts, class_name)            # НОВОЕ — фильтр 9
texts = filter_truncated(texts, class_name)             # НОВОЕ — фильтр 10
texts = filter_prompt_leak(texts, class_name)
# ...дальше cosine_similarity (фильтр 11)
Также обновить docstring файла:


"""
Фильтры:
1. Точные дубликаты — убираем совпадения с существующими и между собой
2. Короткие тексты — мусор/обрезки
3. Не русский язык — после обратного перевода
4. Вырожденные тексты — повторы слов, бессмыслица
5. Иностранные символы — CJK иероглифы (китайский/японский/корейский)
6. Промпт-утечка — LLM описала задание вместо письма
7. Косинусное сходство — слишком похожие на существующие
8. Сломанные плейсхолдеры — артефакты NLLB (<10>, NEFT, OKPO, ...)
9. Максимальная длина — фильтр галлюцинаций (>5000 симв)
10. Обрезанные на середине — текст не завершён грамматически
"""
1.5 Verification (как проверить что Шаг 1 готов)

cd /Users/kvt/Documents/VKR/code
python3 -c "
from src.augmentation.validation import (
    filter_broken_placeholders, filter_max_length, filter_truncated
)

# Test broken filter
test = ['Нормальный текст с [ORGANIZATION] и [DATE_TIME].', 
        'Сломанный <FD текст с NEFT и <9> здесь.']
res = filter_broken_placeholders(test, 'test')
assert len(res) == 1, f'Expected 1, got {len(res)}'
print('✓ filter_broken_placeholders works')

# Test max_length
test = ['короткий текст.', 'a' * 6000]
res = filter_max_length(test, 'test')
assert len(res) == 1
print('✓ filter_max_length works')

# Test truncated
test = ['Полное письмо. С уважением, [PERSON].', 
        'Обрезанное письмо на сере']
res = filter_truncated(test, 'test')
assert len(res) == 1
print('✓ filter_truncated works')
print('All filter tests passed.')
"
ШАГ 2 — Создать промпт для Stage 2 v2 (light paraphrase)
Файл: prompts/aug_prompts/paraphrase_v2.txt (новый)

Содержимое файла:


Ты помощник в задаче расширения корпуса корпоративной почты для классификатора.

Класс письма: «{class_name}»
Описание класса: {class_description}

Оригинальное письмо:
---
{original_text}
---

ЗАДАЧА: Создай вариант этого письма с НЕЗНАЧИТЕЛЬНЫМИ изменениями, как если бы тот же сотрудник переписал его в другой день.

ТИП ИЗМЕНЕНИЙ (lite paraphrase):
- Заменить 25-40% слов на синонимы (там где уместно для делового стиля)
- Слегка переформулировать предложения (порядок слов, активный/пассивный залог)
- НЕ менять структуру абзацев и порядок информации
- НЕ добавлять и не удалять смысловые блоки

КРИТИЧНО — НЕ ИЗМЕНЯТЬ:
1. Все плейсхолдеры в [QUADRATIC_BRACKETS]: [ORGANIZATION], [DATE_TIME], [DOCUMENT_NUMBER], [PERSON], [LOCATION], [EMAIL] — буквально, без изменений регистра
2. Российские аббревиатуры: ОКПО, ОГРН, ИНН, КПП, БИК, ОКВЭД, ОКАТО — только кириллицей
3. Названия проектов и подразделений — без изменений
4. Числа, даты, артикулы документов — точно как в оригинале

ЗАПРЕТ:
- НЕ использовать латиницу для русских слов (NEFT вместо НЕФТЬ — запрещено)
- НЕ упоминать темы, не относящиеся к классу «{class_name}»
- НЕ добавлять метаинформацию («это вариант письма», «парафраз», «переформулированное»)
- НЕ обрывать письмо на середине слова — всегда заканчивать корректно

ТРЕБОВАНИЯ К ДЛИНЕ:
- 1000-2500 символов (близко к медиане реальных писем = 1288)
- НЕ короче 700 символов
- НЕ длиннее 4000 символов

Выведи ТОЛЬКО текст письма. Без markdown, без кавычек вокруг текста, без префиксов.
ШАГ 3 — Создать промпт для Stage 3 v2 (deep paraphrase)
Файл: prompts/aug_prompts/paraphrase_v3.txt (новый)

Содержимое файла:


Ты помощник в задаче расширения корпуса корпоративной почты для классификатора.

Класс письма: «{class_name}»
Описание класса: {class_description}

Оригинальное письмо:
---
{original_text}
---

ЗАДАЧА: Создай ГЛУБОКО ПЕРЕРАБОТАННЫЙ вариант этого письма, как если бы ДРУГОЙ сотрудник той же компании написал письмо по той же тематике, но со своим стилем.

ТИП ИЗМЕНЕНИЙ (deep paraphrase):
- Изменить структуру: переставить абзацы, преобразовать списки в текст или наоборот
- Сменить ролевую перспективу: если оригинал — приказ, новый текст может быть отчётом о выполнении; если запрос — подтверждение получения; и т.д. (но КЛАСС остаётся тем же)
- Полностью переформулировать предложения, использовать другую лексику
- Добавить или убрать структурные элементы (заголовки, нумерацию, подразделы) — если уместно
- Можно изменить уровень формальности в пределах корпоративного стиля

КРИТИЧНО — НЕ ИЗМЕНЯТЬ:
1. Все плейсхолдеры в [QUADRATIC_BRACKETS]: [ORGANIZATION], [DATE_TIME], [DOCUMENT_NUMBER], [PERSON], [LOCATION], [EMAIL] — буквально
2. Российские аббревиатуры: ОКПО, ОГРН, ИНН, КПП, БИК, ОКВЭД, ОКАТО — только кириллицей
3. Названия проектов и подразделений — без изменений
4. Тематика и адресат должны строго соответствовать классу «{class_name}»

ЗАПРЕТ:
- НЕ использовать латиницу для русских слов (NEFT вместо НЕФТЬ — запрещено)
- НЕ упоминать темы из других классов корпоративной таксономии
- НЕ генерировать слишком похожий на оригинал текст (cosine similarity 0.95+ не нужен)
- НЕ обрывать письмо на середине слова

ТРЕБОВАНИЯ К ДЛИНЕ:
- 1200-3500 символов (старайся попадать в более длинные варианты, чтобы покрывать длинный хвост распределения)
- Желательно встречаются варианты до 3500 символов
- НЕ короче 800 символов

ХОРОШИЙ ВАРИАНТ:
- Smly другой стиль, чем оригинал
- Сохраняет фактическое содержание (что/кому/зачем)
- Семантически близок (той же категории), но текстуально далёк (другие слова, другая структура)

Выведи ТОЛЬКО текст письма. Без markdown, без кавычек, без префиксов.
ШАГ 4 — Создать новый модуль Stage 2 v2
Файл: src/augmentation/stage2_paraphrase_v2.py (новый)

Используй существующий stage2_paraphrase.py как шаблон. Изменения:

Импортировать прежние утилиты из src/augmentation/llm_utils.py и src/augmentation/validation.py
Использовать новый промпт paraphrase_v2.txt
Целевой TARGET_COUNT = 50 (было 35)
LLM-judge порог поднять до 4.0 (если есть в pipeline)
Сохранять в Data/data_after_stage2_v2.csv
Ключевые моменты в коде:


"""
stage2_paraphrase_v2.py — Этап 2 v2: light paraphrase через Qwen2.5-32B

Заменяет старый Stage 2 (paraphrase.txt). Использует новый промпт 
paraphrase_v2.txt с явной защитой плейсхолдеров и контролем длины.

Берёт классы B и C, доводит до 50 примеров (было 35).

Вход:  Data/data_after_stage1.csv
Выход: Data/data_after_stage2_v2.csv
"""

# ... импорты как в оригинале ...

STAGE = 2
TARGET_COUNT = 50           # было 35
MAX_RETRIES = 5
OVERSAMPLE_FACTOR = 5       # генерим в 5x больше чем нужно, потом валидация
PARAPHRASE_PROMPT = "paraphrase_v2.txt"  # НОВЫЙ файл
JUDGE_THRESHOLD = 4.0       # повышенный порог (был 2.5)

def main():
    # ... аналогично stage2_paraphrase.py ...
    # Просто меняется промпт-файл и TARGET_COUNT
    pass
Важно: не удаляй старый stage2_paraphrase.py — оставь его в репо для воспроизводимости. Просто новый модуль рядом.

ШАГ 5 — Создать новый модуль Stage 3 v2
Файл: src/augmentation/stage3_paraphrase_v2.py (новый)

Используй ту же структуру что Stage 2 v2, но:

Промпт: paraphrase_v3.txt
Вход: Data/data_after_stage2.csv
Выход: Data/data_after_stage3.csv
TARGET_COUNT = 50

"""
stage3_paraphrase_v2.py — Этап 3 v2: deep paraphrase через Qwen2.5-32B

Заменяет старый Stage 3 (NLLB back-translation, который ломал плейсхолдеры
и транслитерировал русские слова в латиницу).

Использует новый промпт paraphrase_v3.txt с инструкцией глубокой 
переработки (структура, перспектива) при сохранении класса.

Вход:  Data/data_after_stage2_v2.csv
Выход: Data/data_after_stage3_v2.csv
"""

STAGE = 3
TARGET_COUNT = 50
MAX_RETRIES = 5
OVERSAMPLE_FACTOR = 4
PARAPHRASE_PROMPT = "paraphrase_v3.txt"
JUDGE_THRESHOLD = 4.0

# Дополнительно: можно использовать temperature 0.8 (выше чем у Stage 2 = 0.6),
# чтобы получить большее разнообразие
SAMPLING_TEMP = 0.8
ШАГ 6 — Скрипт запуска полного pipeline
Файл: scripts/run_reaugmentation.py (новый)


"""
run_reaugmentation.py — полный прогон новой аугментации.

Шаги:
1. Stage 2 v2 (light paraphrase): data_after_stage1.csv → data_after_stage2_v2.csv
2. Stage 3 v2 (deep paraphrase): data_after_stage2_v2.csv → data_after_stage3_v2.csv
3. Финальная валидация: проверки на broken, длину, распределение
"""

import sys
from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 1. Запускаем Stage 2 v2
print("=" * 60)
print("STAGE 2 V2: Light paraphrase через Qwen2.5-32B")
print("=" * 60)
from src.augmentation.stage2_paraphrase_v2 import main as run_stage2
run_stage2()

# 2. Запускаем Stage 3 v2
print("=" * 60)
print("STAGE 3 V2: Deep paraphrase через Qwen2.5-32B")
print("=" * 60)
from src.augmentation.stage3_paraphrase_v2 import main as run_stage3
run_stage3()

# 3. Финальная валидация
print("=" * 60)
print("ФИНАЛЬНАЯ ВАЛИДАЦИЯ")
print("=" * 60)

df = pd.read_csv("Data/data_after_stage3_v2.csv")
df_test = pd.read_csv("Data/data_test.csv")

# Проверка 1: нет broken placeholders
broken_re = re.compile(
    r'<\d+>?|<[A-Z]{2,5}\b|\bNEFT\b|\bGAZ\b|\bOKPO\b|\bOGRN\b|\bINN\b|\bKPP\b|\bBIK\b'
)
broken_count = df['text'].str.contains(broken_re, regex=True).sum()
assert broken_count == 0, f"❌ Найдено {broken_count} broken-строк"
print(f"✓ Broken placeholders: 0")

# Проверка 2: каждый класс имеет ≥40 примеров (target 50, допуск -10)
class_counts = df['label'].value_counts()
min_count = class_counts.min()
print(f"✓ Min per-class: {min_count} (target ≥40)")
assert min_count >= 40, f"❌ Класс {class_counts.idxmin()} имеет только {min_count} примеров"

# Проверка 3: распределение длин
aug_median = df['text'].str.len().median()
test_median = df_test['text'].str.len().median()
length_ratio = abs(aug_median - test_median) / test_median
print(f"  Median length: aug={aug_median:.0f}, test={test_median:.0f}, ratio={length_ratio:.2%}")
assert length_ratio < 0.20, f"❌ Расхождение медиан длин >20%"
print(f"✓ Length distribution close to test")

# Проверка 4: нет точных дубликатов с test
test_texts = set(df_test['text'].tolist())
overlap = df[df['text'].isin(test_texts)]
assert len(overlap) == 0, f"❌ Найдено {len(overlap)} дубликатов с test (data leak!)"
print(f"✓ No test data leak")

# Итоговая статистика
print(f"\n=== ИТОГОВАЯ СТАТИСТИКА ===")
print(f"Total rows: {len(df)}")
print(f"Classes: {df['label'].nunique()}")
print(f"Per-class: min={class_counts.min()}, max={class_counts.max()}, mean={class_counts.mean():.0f}")
print(f"Length: median={aug_median:.0f}, p95={df['text'].str.len().quantile(0.95):.0f}")
ШАГ 7 — Запуск pipeline

cd /Users/kvt/Documents/VKR/code

# Запуск полного pipeline (примерно 4-6 часов на A100)
python scripts/run_reaugmentation.py
Должно создаться два новых файла:

Data/data_after_stage2_v2.csv (~1900-2000 строк)
Data/data_after_stage3_v2.csv (~2300-2500 строк)
ШАГ 8 — Запуск Fine-tuning на новых данных
В notebooks/finetune.ipynb, в cell 13, изменить путь к training данным:


# Найти строку, где грузится train data, например в src/finetune/data_loader.py
# или прямо в trainer_base.py — путь к data_after_stage3.csv
# Заменить на data_after_stage3_v2.csv

cfg['train_data'] = 'Data/data_after_stage3_v2.csv'  # было stage3.csv
Конфиг fine-tune не меняем (focal loss + class_weights + val_split=0.10 как было).

Запуск через notebook — те же ячейки, что и раньше. Время ~5h на A100.

ШАГ 9 — Замер метрик и сравнение
После окончания FT в results/finetune_results.csv появится новая строка с run_key=qlora_qwen3_32b (она перезапишет старую — это нормально, у нас идёт прогресс).

Сравнить с baseline:

Baseline	macro_f1	bal_acc	f1_C
Старый focal best	0.541	0.559	0.589
Цель (новые данные)	≥0.56	≥0.57	≥0.60
Critical metric: macro_f1 должен быть ≥0.55. Если меньше — что-то пошло не так с реагментацией, нужен post-mortem.

ШАГ 10 — Verification после FT
Создать diagnostic файл diagnostic_after_reaugmentation.md:


# После реагментации Stage 2 v2 + Stage 3 v2

## Метрики на test
- macro_f1: X.XXX (vs baseline 0.541, Δ +X.XXX)
- bal_acc: X.XXX (vs baseline 0.559, Δ +X.XXX)
- f1_A: X.XXX
- f1_B: X.XXX
- f1_C: X.XXX

## Изменения в данных
- Total rows: 2331 → XXXX
- Broken placeholders: 224 → 0
- Median length: 1027 → XXXX
- p95 length: 2501 → XXXX

## Decision
- [ ] macro_f1 ≥ 0.56 → успех, останавливаем реагментацию
- [ ] macro_f1 0.54-0.56 → частичный успех, нужны дополнительные шаги (contrastive)
- [ ] macro_f1 < 0.54 → провал, post-mortem нужен
Резюме (для Codex)
Задачи по порядку:

✅ Расширить src/augmentation/validation.py — 3 новых фильтра (broken_placeholders, max_length, truncated)
✅ Создать prompts/aug_prompts/paraphrase_v2.txt (light paraphrase)
✅ Создать prompts/aug_prompts/paraphrase_v3.txt (deep paraphrase)
✅ Создать src/augmentation/stage2_paraphrase_v2.py (модуль)
✅ Создать src/augmentation/stage3_paraphrase_v2.py (модуль, заменяет NLLB)
✅ Создать scripts/run_reaugmentation.py (orchestrator)
⏳ Запустить pipeline (~4-6 часов на A100)
⏳ Запустить FT (~5 часов на A100)
⏳ Зафиксировать метрики
Ожидаемый результат: macro_f1 0.541 → 0.56-0.59 (+2-5pt).