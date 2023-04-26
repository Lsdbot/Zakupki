"""
Программа: Получение данных по пути и чтение
Версия: 1.0
"""

import pandas as pd


def extract_vector(string: str) -> list:
    return list(map(float, string[1:-1].split()))


def get_data(file_path: str, vector: str) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param data_path: путь до данных
    :return: датасет
    """
    df = pd.read_csv(file_path, index_col='index')
    df[vector] = df[vector].apply(extract_vector)
    return df
