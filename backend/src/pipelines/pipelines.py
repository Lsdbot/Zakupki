"""
Программа: Сборный конвейер для тренировки модели
Версия: 1.0
"""

import yaml

from ..recommender_train.train import train_model as recommender_train, find_optimal_params as recommender_find_params
from ..win_predictor_train.train import train_model as win_train, find_optimal_params as win_find_params
from ..transform_data.transform import transform_vector, get_supplier_data
from ..data.get_data import get_data_with_vector, get_data_without_vector
from ..upload_data.upload import load_recommend_model, load_model


def pipeline_train_recommender(config_path, supplier_id: int):
    """
    Пайплайн по обучению модели рекомендаций для указанного поставщика.

    Аргументы:
    - config_path (str) - путь до конфигурационного файла, включающего:
        - preprocessing (dict) - параметры предобработки данных, включающие:
            - train_data (str) - путь к файлу с обучающими данными
            - test_data (str) - путь к файлу с тестовыми данными
            - recommender (dict) - параметры модели рекомендаций, включающие:
                - sup_column (str) - название столбца с id поставщика
                - index_column (str) - название столбца с id покупок
        - train (dict) - параметры обучения модели, включающие:
            - vector (str) - название столбца с векторами товаров
            - n_components (int) - количество компонент вектора
            - n_trials (int) - количество запусков оптимизации гиперпараметров
            - N_FOLDS (int) - количество фолдов для кросс-валидации
            - random_state (int) - значение для инициализации генератора случайных чисел
    - supplier_id (str) - id поставщика, для которого производится обучение

    Возвращает обученную модель рекомендаций для указанного поставщика.
    """
    # загрузка конфигурации модели
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preproc = config["preprocessing"]
    train = config["train"]["recommender"]

    train_data = get_data_with_vector(preproc['train_data'], train['vector'])
    test_data = get_data_with_vector(preproc['test_data'], train['vector'])

    test_purchases = test_data[test_data[preproc['recommender']['sup_column']] ==
                               supplier_id][preproc['recommender']['index_column']].tolist()
    train_purchases = train_data[train_data[preproc['recommender']['sup_column']] ==
                                 supplier_id][preproc['recommender']['index_column']].tolist()

    train_data = transform_vector(train_data, preproc['n_components'], train['vector'])
    test_data = transform_vector(test_data, preproc['n_components'], train['vector'])

    train_data, test_data = get_supplier_data(train_data, test_data,
                                              supplier_id, test_purchases,
                                              **preproc['recommender'])

    train_data['target'] = train_data.index.isin(train_purchases).astype(int)

    x_train = train_data[train_data.columns[:-1]]
    y_train = train_data['target']

    study = recommender_find_params(x_train, y_train, n_trials=train['n_trials'],
                                    N_FOLDS=train['N_FOLDS'], random_state=train['random_state'])

    model = recommender_train(x_train, y_train, study)

    load_recommend_model(model, supplier_id, **train)


def pipeline_train_win_predictor(config_path: str) -> None:
    """
    Пайплайн для обучения модели, прогнозирующую победителя

    :param config_path: путь до конфигурационного файла
    :return: None
    """
    # загрузка конфигурации модели
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Получение данных и их предобработка
    preproc = config["preprocessing"]
    train = config["train"]["win_predictor"]

    train_data = get_data_without_vector(preproc['train_data'])

    train_data = train_data.astype(preproc['change_type_columns'])
    train_data = train_data.drop(columns=train['drop_columns'])

    x_train = train_data.drop(train['target'], axis=1)
    y_train = train_data[train['target']]

    # Оптимизация параметров модели
    # study = win_find_params(x_train, y_train, **train)

    with open(train['params']) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # Обучение модели
    model = win_train(x_train, y_train, **params['catboost'])

    load_model(model, **train)
