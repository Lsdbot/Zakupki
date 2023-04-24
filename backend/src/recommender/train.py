"""
Программа: Тренировка данных
Версия: 1.0
"""

import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier

import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def train_model(x_train, y_train, study) -> LGBMClassifier:

    params = study.best_params

    model = LGBMClassifier(n_jobs=-1, **params)
    model.fit(x_train, y_train)

    return model


def objective(trial: optuna.Trial, x: pd.DataFrame, y: pd.Series, **kwargs) -> np.ndarray:
    """
    This function defines the objective function for an Optuna study to tune hyperparameters
    for a LightGBM binary classification model.

    Args:
        trial (optuna.Trial): A trial corresponding to a set of hyperparameters.
        x (pd.DataFrame): The features to be used for training and validation.
        y (pd.Series): The target variable for training and validation.

    Returns:
        float: The mean of the cross-validation AUC-ROC scores for the given set of hyperparameters.
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
    """Find optimal hyperparameters using Optuna.

    Args:
    x_train: Pandas dataframe, training feature matrix.
    y_train: Pandas series, training target vector.
    **kwargs: Dictionary with additional arguments:
        - N_FOLDS: int, number of cross-validation folds.
        - random_state: int, random state for reproducibility.
        - n_trials: int, number of trials in the optimization.

    Returns:
    optuna.Study: Object containing the results of the hyperparameter optimization.
    """

    def func(trial):
        return objective(trial, x_train, y_train, N_FOLDS=kwargs['N_FOLDS'], random_state=kwargs['random_state'])

    study = optuna.create_study(direction="maximize")
    study.optimize(func, n_trials=kwargs['n_trials'], n_jobs=-1)

    return study
