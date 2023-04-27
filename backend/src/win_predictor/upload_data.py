"""
Программа: Сохранение данных
Версия: 1.0
"""

import yaml
import joblib

from .pipeline import pipeline_training


def load_model(config_path: str) -> None:
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

    models = joblib.load(config['train']['win_predictor']['models'])

    models['catboost'] = pipeline_training(config)

    joblib.dump(models, config['train']['win_predictor']['models'])
