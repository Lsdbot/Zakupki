"""
Программа: Сохранение данных
Версия: 1.0
"""

import yaml
import joblib

from .pipeline import pipeline_training


def load_model(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    train = config['train']['win_predictor']

    model = pipeline_training(config)

    joblib.dump(model.params_, train['params'])
    joblib.dump(model, train['model'])
