"""
Программа: Вывод данных
Версия: 1.0
"""

from .pipeline import pipeline_evaluate_recommends, pipeline_evaluate_predicts

import pandas as pd

import yaml

import re


def extract_purchases(string: str) -> list:
    return list(map(int, re.findall(r"'(\d+)'", string)))


def load_recommends(config_path: str, supplier_id: int) -> list:
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return pipeline_evaluate_recommends(config, supplier_id)


def load_predicts(config_path: str) -> list:
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return pipeline_evaluate_predicts(config)


def load_users(config_path: str) -> list:
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    recommender_sub = pd.read_csv(config['preprocessing']['recommend_sub_path'], index_col='index')
    recommender_sub['purchases'] = recommender_sub['purchases'].apply(extract_purchases)

    return list(recommender_sub.index)
