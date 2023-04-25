"""
Программа: Получение данных из файла
Версия: 1.0
"""

import pandas as pd

import re

import joblib

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def extract_purchases(string: str) -> list:
    return list(map(int, re.findall(r"'(\d+)'", string)))


def extract_vector(string: str) -> list:
    return list(map(float, string[1:-1].split()))


def get_data(file_path: str, vector: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, index_col='index')
    df[vector] = df[vector].apply(extract_vector)
    return df


def get_recommend_model(filepath: str, supplier_id: int) -> LGBMClassifier:
    models = joblib.load(filepath)
    return models[supplier_id]


def get_win_submission(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, index_col='index')


def get_win_model(filepath: str) -> CatBoostClassifier:
    return joblib.load(filepath)['catboost']
