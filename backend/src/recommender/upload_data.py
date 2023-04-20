"""
Программа: Сохранение данных
Версия: 1.0
"""

import yaml
import joblib

from pipeline import pipeline_training


def load_model(config_path, supplier_id):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    train = config['train']['recommender']

    with open(train['params']) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    with open(train['models']) as file:
        models = yaml.load(file, Loader=yaml.FullLoader)

    model = pipeline_training(config, supplier_id)

    params[supplier_id] = model.params_
    models[supplier_id] = model

    joblib.dump(params, train['params'])
    joblib.dump(models, train['models'])
