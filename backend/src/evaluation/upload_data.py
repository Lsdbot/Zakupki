"""
Программа: Вывод данных
Версия: 1.0
"""

from .pipeline import pipeline_evaluate

import yaml


def load_data(config_path: str, supplier_id: int) -> list:
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return pipeline_evaluate(config, supplier_id)