"""
Программа: Получение данных из файла
Версия: 1.0
"""

import pandas as pd
import joblib
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


def extract_vector(string: str) -> list:
    """
    Преобразует строку с вектором в список чисел с плавающей запятой.

    Аргументы:
        string (str): Строка, содержащая вектор, разделенный запятыми и заключенный в квадратные скобки.

    Возвращает:
        list: Список чисел с плавающей запятой, представляющий вектор из строки.
    """
    return list(map(float, string[1:-1].split()))


def extract_purchases(string: str) -> list:
    """
    Преобразует строку с покупками в список целых чисел.

    Аргументы:
        string (str): Строка, содержащая покупки, разделенные запятыми и заключенные в квадратные скобки.

    Возвращает:
        list: Список целых чисел, представляющий покупки из строки.
    """
    return list(map(int, string[1:-1].replace(',', ' ').split()))


def get_data(file_path: str) -> pd.DataFrame:
    """
    Читает файл CSV из указанного пути и возвращает pandas DataFrame.

    Аргументы:
        file_path (str): Строка, представляющая путь к файлу CSV, который нужно прочитать.

    Возвращает:
        pd.DataFrame: Pandas DataFrame, содержащий данные, прочитанные из файла CSV.
    """
    return pd.read_csv(file_path, sep=';')


def get_data_without_vector(file_path: str) -> pd.DataFrame:
    """
    Читает предобработанный файл CSV из указанного пути и возвращает pandas DataFrame.

    Аргументы:
        file_path (str): Строка, представляющая путь к файлу CSV, который нужно прочитать.

    Возвращает:
        pd.DataFrame: Pandas DataFrame, содержащий данные, прочитанные из файла CSV.
    """
    return pd.read_csv(file_path, index_col='index')


def get_data_with_vector(file_path: str, vector: str) -> pd.DataFrame:
    """
    Читает файл CSV из указанного пути и возвращает pandas DataFrame с добавленным вектором.

    Аргументы:
        file_path (str): Строка, представляющая путь к файлу CSV, который нужно прочитать.
        vector (str): Строка, представляющая имя столбца в файле, содержащего векторы.

    Возвращает:
        pd.DataFrame: Pandas DataFrame, содержащий данные, прочитанные из файла CSV и добавленный столбец с векторами.
    """
    df = pd.read_csv(file_path, index_col='index')
    df[vector] = df[vector].apply(extract_vector)
    return df


def get_recommender_model(filepath: str, supplier_id: int) -> LGBMClassifier:
    """
    Возвращает модель машинного обучения, обученную для определенного поставщика товаров.

    Аргументы:
        filepath (str): Путь к файлу, содержащему сохраненные модели машинного обучения.
        supplier_id (int): Идентификатор поставщика, для которого требуется вернуть модель.

    Возвращает:
        LGBMClassifier: Объект модели машинного обучения LGBMClassifier.
    """
    models = joblib.load(filepath)
    return models[supplier_id]


def get_win_predictor_model(filepath: str) -> CatBoostClassifier:
    """
    Возвращает модель машинного обучения, предсказывающую победителя в аукционе.

    Аргументы:
        filepath (str): Путь к файлу, содержащему сохраненную модель машинного обучения.

    Возвращает:
        CatBoostClassifier: Объект модели машинного обучения CatBoostClassifier.
    """
    return joblib.load(filepath)['catboost']


def get_submission(file_path: str, purchases: str) -> pd.DataFrame:
    """
    Читает файл CSV из указанного пути и возвращает pandas DataFrame, содержащий закупки поставщиков.

    Аргументы:
        file_path (str): Строка, представляющая путь к файлу CSV, который нужно прочитать.
        purchases (str): Строка, представляющая имя столбца в файле, содержащего закупки поставщиков.

    Возвращает:
        pd.DataFrame: Pandas DataFrame, содержащий данные, прочитанные из файла CSV со списком закупок поставщиков.
    """
    df = pd.read_csv(file_path, index_col='index')
    df[purchases] = df[purchases].apply(extract_purchases)
    return df


def get_users(config_path: str) -> list:
    """
    Загружает конфигурационный файл, применяет функцию extract_purchases к столбцу 'purchases'
    датафрейма, считываемого из файла по пути config['preprocessing']['recommend_sub_path'],
    и возвращает список всех id поставщиков.

    :param config_path: путь к конфигурационному файлу
    :return: список всех id поставщиков

    Пример использования:
    users = load_users('config.yaml')
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    recommender_sub = pd.read_csv(config['preprocessing']['recommend_sub_path'], index_col='index')
    recommender_sub['purchases'] = recommender_sub['purchases'].apply(extract_purchases)

    return list(recommender_sub.index)
