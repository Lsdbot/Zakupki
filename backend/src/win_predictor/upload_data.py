"""
Программа: Сохранение данных
Версия: 1.0
"""

import yaml

from typing import Dict

from .pipeline import pipeline_training


def load_model(config_path: str) -> Dict[str, any]:
    """
    Загружает конфигурацию модели из указанного файла config_path,
    запускает pipeline_training для обучения модели и возвращает параметры обученной модели.

    Args:
        config_path: Путь к файлу с конфигурацией модели в формате yaml.

    Returns:
        Словарь параметров обученной модели.
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    model = pipeline_training(config)

    return model.get_params()
