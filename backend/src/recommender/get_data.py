"""
Программа: Получение данных из файла
Версия: 1.0
"""

import pandas as pd


def extract_vector(string):
    return list(map(float, string[1:-1].split()))

def extract_purchases(string):
    return list(map(int, string[1:-1].replace(',', ' ').split()))

def get_data(file_path, vector):
    df = pd.read_csv(file_path, index_col='index')
    df[vector] = df[vector].apply(extract_vector)

    return df

def get_submission(file_path, purchases):
    df = pd.read_csv(file_path, index_col='index')
    df[purchases] = df[purchases].apply(extract_purchases)

    return df