"""
Программа: Получение предсказания на основе обученной модели
Версия: 1.0
"""

import yaml
from pandas import DataFrame

from ..data.get_data import get_recommender_model, get_win_predictor_model, \
    get_data_with_vector, get_data_without_vector
from ..transform_data.transform import transform_vector, get_supplier_data, extract_month, \
    extract_reg_code, extract_purchase_size


def generate_features(df_train: DataFrame, df_evaluate: DataFrame, **kwargs) -> DataFrame:
    """
    Функция для преобразования данных тестового наборов с помощью инженерии признаков.

    Аргументы:
    - df_train (pd.DataFrame): Данные для тренировки.
    - df_evaluate (pd.DataFrame): Данные для получения предсказаний.

    Возвращаемое значение:
    - DataFrame: Обработанные данные для получения предсказаний.
    """

    # Извлечение месяца из даты
    df_evaluate['month'] = extract_month(df_evaluate, kwargs['date'])

    # Извлечение кода региона и ОКПД2
    df_evaluate['reg_code'] = extract_reg_code(df_evaluate, kwargs['region'], kwargs['okpd2'])

    # Извлечение размера покупки
    purchase_s = kwargs['purchase_size']
    df_evaluate = extract_purchase_size(df_evaluate, purchase_s['group'], purchase_s['values'],
                                        purchase_s['name'], purchase_s['on_col'])

    # Извлечение флага, информирующего о предыдущем контакте поставщика и покупателя
    flag = kwargs['flag']
    df_evaluate = df_evaluate.merge(df_train[flag['train_columns']].groupby(flag['group']).tail(1),
                                    on=flag['on_col'], how='left').fillna(0)

    # Извлечение кол-ва уникальных кодов ОКПД2
    uniq_okpd2 = kwargs['uniq_okpd2']
    df_evaluate = df_evaluate.merge(df_train[[uniq_okpd2['group'], uniq_okpd2['name']]].groupby(
        uniq_okpd2['group']).tail(1), on=uniq_okpd2['on_col'], how='left').fillna(1)

    # Удаление ненужных столбцов
    df_evaluate = df_evaluate.drop(columns=kwargs['drop_columns'])

    return df_evaluate


def pipeline_evaluate_recommends(config_path: str, supplier_id: int) -> list:
    """
    Выполняет оценку рекомендательной системы на тестовой выборке для заданного поставщика.

    Аргументы:
    - config_path (str): путь до конфигурационного файла.
    - supplier_id (int): идентификатор поставщика.

    Возвращаемое значение:
    - list: список индексов тестовых записей, рекомендованных для заданного поставщика.
    """
    # загрузка конфигурации модели
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # извлекаем параметры предобработки, обучения и оценки
    preproc = config['preprocessing']
    train_recommender = config['train']['recommender']
    evaluate = config['evaluate']

    # получаем данные для обучения и оценки
    train_data = get_data_with_vector(preproc['train_data'], train_recommender['vector'])
    evaluate_data = get_data_with_vector(evaluate['evaluate_data'], train_recommender['vector'])

    # генерируем признаки для тестовой выборки
    evaluate_data = generate_features(train_data, evaluate_data, **preproc)

    # получаем список закупок для заданного поставщика
    test_purchases = evaluate_data[evaluate_data[preproc['recommender']['sup_column']] ==
                                   supplier_id][preproc['recommender']['index_column']]

    # преобразуем данные, указав размерность вектора токенов
    evaluate_data = transform_vector(evaluate_data, n_components=preproc['n_components'],
                                     vector_column=train_recommender['vector'])

    # получаем данные только для заданного поставщика
    evaluate_sup_data = get_supplier_data(train_data, evaluate_data, supplier_id,
                                          test_purchases, **train_recommender)[1]

    # загружаем модель для заданного поставщика и делаем предсказание
    model = get_recommender_model(train_recommender['models'], supplier_id)
    y_pred = model.predict(evaluate_sup_data)

    # возвращаем список индексов записей, рекомендованных для заданного поставщика
    return evaluate_sup_data[y_pred == 1].index.tolist()


def pipeline_evaluate_predicts(config_path: str) -> list:
    """
    Выполняет оценку модели предсказания победителя на тестовой выборке.

    Аргументы:
    - config_path (str): путь до конфигурационного файла.

    Возвращаемое значение:
    - list:
    """
    # загрузка конфигурации модели
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # извлекаем параметры предобработки, обучения и оценки
    preproc = config['preprocessing']
    train_win_predictor = config['train']['win_predictor']
    evaluate = config['evaluate']

    # получаем данные для обучения и оценки
    train_data = get_data_without_vector(preproc['train_data'])
    evaluate_data = get_data_without_vector(evaluate['evaluate_data'])

    # генерируем признаки для тестовой выборкиpipeline_evaluate_predi
    evaluate_data = generate_features(train_data, evaluate_data, **preproc)

    # преобразуем типы данных и удаляем столбцы
    evaluate_data = evaluate_data.astype(preproc['change_type_columns'])
    evaluate_data = evaluate_data.drop(columns=train_win_predictor['drop_columns'])

    evaluate_data = evaluate_data.drop('is_winner', axis=1)

    # загружаем модель и предсказываем, победит ли поставщик в закупке
    model = get_win_predictor_model(train_win_predictor['models'])
    y_pred = model.predict(evaluate_data)

    return list(y_pred)
