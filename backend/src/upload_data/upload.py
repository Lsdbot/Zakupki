"""
Программа: Сохранение моделей и их параметров
Версия: 1.0
"""

import joblib
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def load_recommend_model(model: LGBMClassifier, supplier_id: int, **kwargs) -> None:
    """
    Cораняет обученную рекомендательную модель и ее параметры для конкретного поставщика.

    Аргументы:
    - model (LGBMClassifier): обученная модель
    - supplier_id (int): идентификатор поставщика.
    - kwargs (dict):
        - models (str): путь к файлу конфигурации, содержащему модели.
        - params (str): путь к файлу конфигурации, содержащему параметы моделей.

    Возвращаемое значение:
    - None.
    """
    # загрузка моделей и их параметров
    with open(kwargs['params']) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    models = joblib.load(kwargs['models'])

    # загружаем модель и параметры в словарь по индексу supplier_id
    models[supplier_id] = model
    params[supplier_id] = model.get_params()

    # сохраняем словарь моделей и параметров
    joblib.dump(models, kwargs['models'])
    with open(kwargs['params'], 'w') as f:
        yaml.dump(params, f)


def load_model(model: CatBoostClassifier, **kwargs) -> None:
    """
    Cохраняет обученную модель, предсказывающую победителя, и ее параметры.

    Аргументы:
    - model (СatBoostClassifier): обученная модель
    - kwargs (dict):
        - models (str): путь к файлу конфигурации, содержащему модели.
        - params (str): путь к файлу конфигурации, содержащему параметы моделей.

    Возвращаемое значение:
    - None.
    """
    # загрузка моделей и их параметров
    with open(kwargs['params']) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    models = joblib.load(kwargs['models'])

    # загружаем модель catboost и ее параметры
    models['catboost'] = model
    params['catboost'] = model.get_params()

    # сохраняем словарь моделей и параметров
    joblib.dump(models, kwargs['models'])
    with open(kwargs['params'], 'w') as f:
        yaml.dump(params, f)


def load_data(df_train: pd.DataFrame, df_test: pd.DataFrame, df_recommend_submission: pd.DataFrame,
              df_evaluate: pd.DataFrame, **kwargs) -> None:
    """
    Загрузка данных из файла конфигурации и сохранение обработанных данных в CSV-файлы.

    Args:
    - df_train: DataFrame с тренировочными данными.
    - df_test: DataFrame с тестовыми данными.
    - df_recommend_submission: DataFrame с данными для получения предсказаний.
    - df_evaluate: DataFrame с данными для оценки модели.
    - **kwargs: Аргументы для функции. Содержит:
        - train_data (str): Путь тренировочных предобработанных данных.
        - test_data (str): Путь тестовых предобработанных данных.
        - recommend_sub_path (str): Путь файла, содержащего поставщиков и их закупке из тестовых данных
        - evaluate_data (str): Путь данных для получения предсказаний

    Returns:
    None
    """

    # Сохранение данных в CSV-файлы
    df_train.to_csv(kwargs['train_data'], index_label='index')
    df_test.to_csv(kwargs['test_data'], index_label='index')
    df_recommend_submission.to_csv(kwargs['recommend_sub_path'], index_label='index')
    df_evaluate.to_csv(kwargs['evaluate_data'], index_label='index')
