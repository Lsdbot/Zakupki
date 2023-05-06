"""
Программа: Получение данных по пути и чтение
Версия: 1.0
"""

import pandas as pd


def extract_vector(string: str) -> list:
    """
    Преобразует строку с вектором в список чисел с плавающей запятой.

    Аргументы:
        string (str): Строка, содержащая вектор, разделенный запятыми и заключенный в квадратные скобки.

    Возвращает:
        list: Список чисел с плавающей запятой, представляющий вектор из строки.
    """
    return list(map(float, string[1:-1].split()))


def get_data(file_path: str, vector: str) -> pd.DataFrame:
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
