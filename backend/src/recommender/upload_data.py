"""
Программа: Сохранение данных
Версия: 1.0
"""

import yaml
import joblib

from pipeline import pipeline_training


def load_models(config_path, params_path, models_path, supplier_id):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    with open(models_path) as file:
        models = yaml.load(file, Loader=yaml.FullLoader)

    model = pipeline_training(config, supplier_id)

    params[supplier_id] = model.params_
    models[supplier_id] = model

    joblib.dump(params, params_path)
    joblib.dump(models, models_path)