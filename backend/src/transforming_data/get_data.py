"""
Программа: Получение данных из файла
Версия: 1.0
"""

import pandas as pd

def get_data(file_path):
    return pd.read_csv(file_path, sep=';')