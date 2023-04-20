"""
Программа: Получение данных из файла
Версия: 1.0
"""

import pandas as pd


def get_data(filepath):
    return pd.read_csv(filepath, index_col='index')
