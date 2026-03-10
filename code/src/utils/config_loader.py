"""
config_loader.py — Загрузка конфигурации LLM-модели из JSON-файла

Каждая модель в проекте описывается своим JSON-конфигом в папке configs/.
Этот модуль читает конфиг и отдаёт удобную структуру с именем модели,
параметрами генерации и именем файла промпта.

Зачем отдельный модуль: чтобы не дублировать парсинг конфига в каждом этапе
аугментации — все этапы пользуются одной и той же функцией.
"""

import json
from pathlib import Path


# Обязательные поля в конфиге — если чего-то нет, лучше упасть сразу,
# чем потом ловить непонятные ошибки при генерации
REQUIRED_FIELDS = ["model_name", "generation_params", "prompt_template"]


def load_model_config(config_path: str) -> dict:
    """
    Читает JSON-конфиг модели и возвращает словарь с тремя ключами:
    model_name, generation_params, prompt_template.

    Проверяет, что файл существует и содержит все обязательные поля.
    Если prompt_template указан без пути — подставляет папку prompts/.

    Аргументы:
        config_path: путь до JSON-файла конфига (например, 'configs/model_qwen.json')

    Возвращает:
        Словарь с ключами:
            - model_name (str): имя модели на HuggingFace
            - generation_params (dict): параметры для model.generate()
            - prompt_template (str): полный путь до файла промпта
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Конфиг не найден: {config_path}. "
            f"Проверь, что файл лежит в папке configs/"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Проверяем, что все нужные поля на месте
    missing = [field for field in REQUIRED_FIELDS if field not in config]
    if missing:
        raise KeyError(
            f"В конфиге {config_path} не хватает полей: {', '.join(missing)}. "
            f"Сверься с примером в CLAUDE.md"
        )

    # Если в prompt_template указано просто имя файла (без пути),
    # считаем, что он лежит в prompts/ относительно корня проекта
    prompt_template = config["prompt_template"]
    if not Path(prompt_template).is_absolute() and "/" not in prompt_template:
        project_root = config_path.parent.parent
        prompt_template = str(project_root / "prompts" / prompt_template)

    # Возвращаем весь конфиг целиком — разные этапы могут использовать
    # свои дополнительные поля (system_prompt, use_unsloth, paraphrase_template и т.д.),
    # и терять их нельзя
    config["prompt_template"] = prompt_template
    return config


def load_prompt(prompt_path: str) -> str:
    """
    Читает текст промпта из файла.

    Промпты хранятся в папке prompts/ как обычные текстовые файлы.
    Функция просто читает содержимое и возвращает строку — без магии.

    Аргументы:
        prompt_path: путь до файла промпта

    Возвращает:
        Текст промпта как строку
    """
    prompt_path = Path(prompt_path)

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Файл промпта не найден: {prompt_path}. "
            f"Проверь папку prompts/ и поле prompt_template в конфиге"
        )

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()
