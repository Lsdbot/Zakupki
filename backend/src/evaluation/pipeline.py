"""
Программа: Пайплайн по получению предсказаний на основе обученных моделей
Версия: 1.0
"""

from .get_data import get_data, get_recommend_model, get_win_model
from .evaluate import get_supplier_data, generate_features, transform_vector


def pipeline_evaluate_recommends(config: dict, supplier_id: int) -> list:
    """
    Выполняет оценку рекомендательной системы на тестовой выборке для заданного поставщика.

    Аргументы:
    - config (dict): словарь с параметрами модели и оценки.
    - supplier_id (int): идентификатор поставщика.

    Возвращаемое значение:
    - list: список индексов тестовых записей, рекомендованных для заданного поставщика.
    """

    # извлекаем параметры предобработки, обучения и оценки
    preproc = config['preprocessing']
    train_recommender = config['train']['recommender']
    evaluate = config['evaluate']

    # получаем данные для обучения и оценки
    train_data = get_data(preproc['train_data'], train_recommender['vector'])
    evaluate_data = get_data(evaluate['evaluate_data'], train_recommender['vector'])

    # генерируем признаки для тестовой выборки
    evaluate_data = generate_features(train_data, evaluate_data, preproc)

    # получаем список закупок для заданного поставщика
    test_purchases = evaluate_data[evaluate_data[preproc['recommender']['sup_column']] ==
                                   supplier_id][preproc['recommender']['index_column']]

    # преобразуем данные, указав размерность вектора токенов
    evaluate_data = transform_vector(evaluate_data, n_components=preproc['n_components'],
                                     vector=train_recommender['vector'])

    # получаем данные только для заданного поставщика
    evaluate_sup_data = get_supplier_data(train_data, evaluate_data, supplier_id, test_purchases, **train_recommender)

    # загружаем модель для заданного поставщика и делаем предсказание
    model = get_recommend_model(train_recommender['models'], supplier_id)
    y_pred = model.predict(evaluate_sup_data)

    # возвращаем список индексов записей, рекомендованных для заданного поставщика
    return evaluate_sup_data[y_pred == 1].index.tolist()


def pipeline_evaluate_predicts(config: dict) -> list:
    """
    Выполняет оценку модели предсказания победителя на тестовой выборке.

    Аргументы:
    - config (dict): словарь с параметрами модели и оценки.

    Возвращаемое значение:
    - list:
    """
    # извлекаем параметры предобработки, обучения и оценки
    preproc = config['preprocessing']
    train_win_predictor = config['train']['win_predictor']
    train_recommender = config['train']['recommender']
    evaluate = config['evaluate']

    # получаем данные для обучения и оценки
    train_data = get_data(preproc['train_data'], train_recommender['vector'])
    evaluate_data = get_data(evaluate['evaluate_data'], train_recommender['vector'])

    # генерируем признаки для тестовой выборкиpipeline_evaluate_predi
    evaluate_data = generate_features(train_data, evaluate_data, preproc)

    # преобразуем типы данных и удаляем столбцы
    evaluate_data = evaluate_data.astype(preproc['change_type_columns'])
    evaluate_data = evaluate_data.drop(columns=train_win_predictor['drop_columns'])

    evaluate_data = evaluate_data.drop('is_winner', axis=1)

    print(evaluate_data.columns)

    # загружаем модель и предсказываем, победит ли поставщик в закупке
    model = get_win_model(train_win_predictor['models'])
    y_pred = model.predict(evaluate_data)

    return list(y_pred)
