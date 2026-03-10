"""
parse_vivod.py — Парсит вывод этапа 2 из vivod.txt и сохраняет сгенерированные тексты в CSV.

Логика: ищем строки вида '[Этап 2] Класс «...»: добавлено N парафразов',
затем собираем тексты между разделителями 'Пример сгенерированного письма:' + '---'.
Каждому тексту присваиваем label из заголовка блока.
"""

import re
import csv
from pathlib import Path


def parse_output(text: str) -> list[dict]:
    """Парсит вывод этапа 2 и возвращает список {text, label}."""
    records = []

    # Разбиваем на блоки по классам
    # Паттерн: [Этап 2] Класс «...»: добавлено N парафразов
    class_pattern = re.compile(
        r'\[Этап 2\] Класс [«"](.*?)[»"]: добавлено (\d+) парафразов'
    )

    # Разделитель между письмами
    separator = "Пример сгенерированного письма:\n" + "-" * 50

    # Находим все позиции блоков классов
    class_matches = list(class_pattern.finditer(text))

    for i, match in enumerate(class_matches):
        class_name = match.group(1)
        n_added = int(match.group(2))

        # Границы блока: от текущего совпадения до следующего класса (или конца файла)
        start = match.end()
        if i + 1 < len(class_matches):
            end = class_matches[i + 1].start()
        else:
            end = len(text)

        block = text[start:end]

        # Разбиваем блок по разделителю и собираем тексты
        parts = block.split(separator)

        texts_found = []
        for part in parts[1:]:  # Первый элемент — до первого разделителя (логи)
            # Текст заканчивается перед следующим блоком логов или концом
            # Обрезаем лог-строки в конце (начинаются с [Этап 2], [Валидация], [Попытка и т.д.)
            cleaned = part.strip()
            if cleaned:
                texts_found.append(cleaned)

        for t in texts_found:
            records.append({"text": t, "label": class_name})

        print(f"Класс «{class_name}»: ожидалось {n_added}, найдено {len(texts_found)} текстов")

    return records


def main():
    vivod_path = Path(__file__).parent / "vivod.txt"
    output_path = Path(__file__).parent / "Data" / "stage2_from_vivod.csv"

    print(f"Читаю {vivod_path}...")
    text = vivod_path.read_text(encoding="utf-8")

    records = parse_output(text)

    print(f"\nВсего извлечено {len(records)} текстов")

    # Сохраняем
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Сохранено в {output_path}")

    # Статистика по классам
    from collections import Counter
    counts = Counter(r["label"] for r in records)
    print("\nПо классам:")
    for name, count in sorted(counts.items()):
        print(f"  «{name}»: {count}")


if __name__ == "__main__":
    main()
