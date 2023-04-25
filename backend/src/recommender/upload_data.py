"""
Программа: Сохранение данных
Версия: 1.0
"""

import yaml
import joblib

from .pipeline import pipeline_training


def load_model(config_path: str, supplier_id: int) -> None:
    """
    Загружает предварительно обученную модель машинного обучения для конкретного поставщика из файла конфигурации.

    Аргументы:
    - config_path (строка): путь к файлу конфигурации, содержащему параметры модели.
    - supplier_id (строка): идентификатор поставщика, для которого загружается модель.

    Возвращаемое значение:
    - Нет возвращаемого значения, но функция сохраняет модель и параметры обучения в файлы, если они были изменены.
    """

    # загрузка конфигурации модели
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # получение параметров модели из конфигурации
    train = config['train']['recommender']

    # загрузка конфигурации параметров моделей
    with open(train['params']) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # загрузка моделей из файла
    models = joblib.load(train['models'])

    # если модели для данного поставщика нет в файле, то обучаем новую и сохраняем ее
    if supplier_id not in models:
        model = pipeline_training(config, supplier_id)
        params[supplier_id] = model.get_params()
        models[supplier_id] = model

        # сохраняем новые параметры и модели в файлы
        with open(train['params'], 'w') as f:
            yaml.dump(params, f)
        joblib.dump(models, train['models'])


