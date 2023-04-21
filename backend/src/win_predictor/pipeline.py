"""
Программа: Сборный конвейер для тренировки модели
Версия: 1.0
"""

from .get_data import get_data

from .train import find_optimal_params, train_model


def pipeline_training(config):

    preproc = config["preprocessing"]
    train = config["train"]["win_predictor"]

    train_data = get_data(preproc['train_data'])

    train_data = train_data.astype(preproc['change_type_columns'])
    train_data = train_data.drop(columns=train['drop_columns'])

    x_train = train_data.drop(train['target'], axis=1)
    y_train = train_data[train['target']]

    study = find_optimal_params(x_train, y_train, **train)

    model = train_model(x_train, y_train, train['cat_features'],
                        **study.best_params)

    return model
