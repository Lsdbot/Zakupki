"""
Программа: Тренировка прогнозирующей победителя модели
Версия: 1.0
"""

import pandas as pd
import numpy as np

from catboost import CatBoostClassifier

import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def train_model(x_train: pd.DataFrame, y_train: pd.Series, **params) -> CatBoostClassifier:
    """
    Обучает модель CatBoost с помощью заданных параметров.

    Аргументы:
        x_train: Обучающие данные в виде таблицы pandas DataFrame;
        y_train: Метки классов для обучающих данных в виде pandas Series;
        cat_features: Список индексов категориальных признаков в x_train;
        params: Словарь параметров для модели CatBoost.

    Возвращает:
        CatBoostClassifier: Обученную модель CatBoostClassifier.
    """

    # Вычисляем соотношение классов
    ratio = y_train[y_train == 0].shape[0] / y_train[y_train == 1].shape[0]

    # Создаем модель CatBoostClassifier
    model = CatBoostClassifier(**params)

    # Обучаем модель на обучающих данных
    model.fit(x_train, y_train, verbose=0)

    # Возвращаем обученную модель
    return model


def objective(trial: optuna.Trial, x: pd.DataFrame, y: pd.Series, **kwargs) -> np.ndarray:
    """
    Функция для оптимизации параметров модели CatBoostClassifier с помощью библиотеки Optuna.

    Аргументы:
        trial (optuna.Trial): Объект для оптимизации параметров модели.
        x (pd.DataFrame): Признаки для обучения модели.
        y (pd.Series): Целевая переменная для обучения модели.
        kwargs: Дополнительные аргументы, включая число разбиений для кросс-валидации, категориальные признаки
            и случайное состояние.

    Возвращает:
        np.ndarray: Среднее значение ROC AUC для кросс-валидации по заданному числу разбиений.

    """
    # Определяем параметры модели
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [1000]),
        # 'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.0787449098272658]),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0001, 100),
        'random_strength': trial.suggest_float('random_strength', 10, 50),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 100),
        'border_count': trial.suggest_categorical('border_count', [128]),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'random_state': kwargs['random_state']
    }

    # Инициализируем массив для предсказаний на каждом фолде
    cv_pred = np.empty(kwargs['N_FOLDS'])

    # Создаем разбиения для кросс-валидации
    cv = StratifiedKFold(n_splits=kwargs['N_FOLDS'], shuffle=True, random_state=kwargs['random_state'])

    # Обучаем модель на каждом фолде
    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        x_train_, x_val_ = x.iloc[train_idx], x.iloc[test_idx]
        y_train_, y_val_ = y.iloc[train_idx], y.iloc[test_idx]

        # Определяем отношение числа объектов первого класса к числу объектов второго класса
        ratio = y_train_[y_train_ == 0].shape[0] / \
            y_train_[y_train_ == 1].shape[0]

        model = CatBoostClassifier(
            scale_pos_weight=ratio,
            cat_features=kwargs['cat_features'],
            **params
        )
        model.fit(x_train_, y_train_,
                  eval_set=[(x_val_, y_val_)],
                  early_stopping_rounds=100,
                  verbose=0)

        y_proba = model.predict_proba(x_val_)[:, 1]

        cv_pred[fold] = roc_auc_score(y_val_, y_proba)

    return np.mean(cv_pred)


def find_optimal_params(x_train: pd.DataFrame, y_train: pd.Series,
                        **kwargs: dict) -> optuna.Study:
    """
    Поиск оптимальных параметров модели с помощью оптимизации гиперпараметров с помощью библиотеки Optuna.

    Аргументы:
        x_train: Обучающие данные в виде таблицы pandas DataFrame;
        y_train: Метки классов для обучающих данных в виде pandas Series;
        kwargs: Словарь с параметрами для кросс-валидации и поиска гиперпараметров.

    Возвращает:
        study: Объект класса optuna.Study со списком найденных оптимальных параметров модели.
    """

    # Определяем функцию для оптимизации
    func = lambda trial: objective(trial, x_train, y_train,
                                   N_FOLDS=kwargs['N_FOLDS'],
                                   random_state=kwargs['random_state'],
                                   cat_features=kwargs['cat_features'])

    # Создаем объект Study и запускаем оптимизацию
    study = optuna.create_study(direction="maximize")
    study.optimize(func, show_progress_bar=True, n_trials=kwargs['n_trials'], n_jobs=6)

    # Возвращаем объект Study с найденными оптимальными параметрами
    return study
