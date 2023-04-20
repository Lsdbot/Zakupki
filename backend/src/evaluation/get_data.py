"""
Программа: Получение данных из файла
Версия: 1.0
"""

import pandas as pd

import joblib


def extract_purchases(string):
    return list(map(int, string[1:-1].replace(',', ' ').split()))


def extract_vector(string):
    return list(map(float, string[1:-1].split()))


def get_data(file_path, vector):
    df = pd.read_csv(file_path, index_col='index')
    return df[vector].apply(extract_vector)


def get_recommender_submission(filepath, vector):
    recommender_sub = pd.read_csv(filepath, index_col='index')
    return recommender_sub[vector].apply(extract_purchases)


def get_win_submission(filepath):
    return pd.read_csv(filepath, index_col='index')


def get_win_model(filepath):
    return joblib.load(filepath)


def get_recommend_model(filepath, supplier_id):
    models = joblib.load(filepath)
    return models[supplier_id]
