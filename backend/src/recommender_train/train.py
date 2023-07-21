"""
Программа: Тренировка рекомендательных моделей
Версия: 1.0
"""

import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier

import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def train_model(x_train, y_train, study) -> LGBMClassifier:
    """
    Обучает модель LightGBM с оптимальными гиперпараметрами.

    Аргументы:
        x_train (np.ndarray): Признаки обучающих данных.
        y_train (np.ndarray): Целевая переменная обучающих данных.
        study (optuna.study.Study): Объект исследования Optuna с лучшими гиперпараметрами.

    Возвращает:
        lightgbm.sklearn.LGBMClassifier: Обученная модель LightGBM с лучшими гиперпараметрами.
    """
    params = study.best_params

    model = LGBMClassifier(n_jobs=-1, **params)
    model.fit(x_train, y_train)

    return model


def objective(trial: optuna.Trial, x: pd.DataFrame, y: pd.Series, **kwargs) -> np.ndarray:
    """
    Эта функция определяет целевую функцию для исследования Optuna с настройкой гиперпараметров
    для модели бинарной классификации LightGBM.

    Аргументы:
        trial (optuna.Trial): Проба, соответствующая набору гиперпараметров.
        x (pd.DataFrame): Признаки, используемые для обучения и валидации.
        y (pd.Series): Целевая переменная для обучения и валидации.

    Возвращает:
        float: Среднее значение метрики AUC-ROC кросс-валидации для заданного набора гиперпараметров.
    """

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 40, 400, step=20),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 10),
        'max_bin': trial.suggest_int('max_bin', 10, 120, step=10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 500, step=20),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 100, 5000, step=100),
        'lambda_l1': trial.suggest_int('lambda_l1', 0, 100),
        'lambda_l2': trial.suggest_int('lambda_l2', 0, 100),
        'min_split_gain': trial.suggest_float('min_split_gain', 0, 0.1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 2, 6),
        'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
        'objective': 'binary',
        'metric': 'auc',
        'random_state': kwargs['random_state'],
        'class_weight': 'balanced'
    }

    cv_pred = np.empty(kwargs['N_FOLDS'])
    cv = StratifiedKFold(n_splits=kwargs['N_FOLDS'], shuffle=True, random_state=kwargs['random_state'])

    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        x_train_, x_val_ = x.iloc[train_idx], x.iloc[test_idx]
        y_train_, y_val_ = y.iloc[train_idx], y.iloc[test_idx]

        pruning = optuna.integration.LightGBMPruningCallback(trial, 'auc')

        model = LGBMClassifier(
            n_jobs=-1,
            **params
        )
        model.fit(x_train_, y_train_,
                  eval_metric='auc',
                  eval_set=[(x_val_, y_val_)],
                  early_stopping_rounds=100,
                  callbacks=[pruning],
                  verbose=-1)

        y_proba = model.predict_proba(x_val_)[:, 1]

        cv_pred[fold] = roc_auc_score(y_val_, y_proba)

    return np.mean(cv_pred)


def find_optimal_params(x_train: pd.DataFrame, y_train: pd.Series, **kwargs: dict) -> optuna.Study:
    """
    Находит оптимальные гиперпараметры с помощью Optuna.

    Аргументы:
        x_train: Pandas DataFrame, матрица признаков обучающих данных.
        y_train: Pandas Series, вектор целевой переменной обучающих данных.
        **kwargs: Словарь с дополнительными аргументами:
            - N_FOLDS: int, количество складок для кросс-валидации.
            - random_state: int, случайное значение для воспроизводимости.
            - n_trials: int, количество итераций оптимизации.

    Возвращает:
        optuna.Study: Объект, содержащий результаты оптимизации гиперпараметров.
    """

    def func(trial):
        return objective(trial, x_train, y_train, N_FOLDS=kwargs['N_FOLDS'], random_state=kwargs['random_state'])

    study = optuna.create_study(direction="maximize")
    study.optimize(func, n_trials=kwargs['n_trials'], n_jobs=-1)

    return study
