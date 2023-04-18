"""
Программа: Тренировка данных
Версия: 1.0
"""

import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier

import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, \
                            recall_score, f1_score, log_loss


def get_metrics(y_test, y_pred, y_score) -> dict:
    """Метрики для задачи классификации"""
    metrics = {}

    try:
        metrics['ROC_AUC'] = roc_auc_score(y_test, y_score[:, 1])
        metrics['Precision'] = precision_score(y_test, y_pred, zero_division=0)
        metrics['Recall'] = recall_score(y_test, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
        metrics['Logloss'] = log_loss(y_test, y_score)

    except ValueError:
        metrics['ROC_AUC'] = 0
        metrics['Precision'] = 0
        metrics['Recall'] = 0
        metrics['f1'] = 0
        metrics['Logloss'] = 0

    return metrics


def supplier_data(df_train, df_submission, sup, **kwargs):

    unique_reg_okpd = df_train[df_train[kwargs['suppplier_column']] == sup][kwargs['filter_column']].unique()

    # фильтруем train на основе уникальных reg_code поставщиков
    df_sup_train = df_train[df_train['filter_column'].isin(unique_reg_okpd)]

    # удаляем ненужные для системы рекомендаций столбцы и дубликаты
    df_sup_train = df_sup_train.drop(columns=kwargs['drop_columns_recommender']) \
        .drop_duplicates()

    df_sup_train = df_sup_train.set_index(kwargs['index_column'])

    # удаляем закупки, которые есть и test, и в train
    df_sup_train = df_sup_train.drop(set(df_submission[sup]).intersection(df_sup_train.index))

    return df_sup_train


def train_model(X, Y, study):

    model = LGBMClassifier(n_jobs=-1, **study)
    model.fit(X, Y)

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
        'n_estimators': trial.suggest_categorical('n_estimators', [300]),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 3),
        'max_bin': trial.suggest_int('max_bin', 0, 120, step=10),
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


def find_optimal_params(x_train, y_train, **kwargs) -> optuna.Study:

    func = lambda trial: objective(trial, x_train, y_train, N_FOLDS=kwargs['N_FOLDS'],
                                   random_state=kwargs['random_state'])

    study = optuna.create_study(direction="maximize")
    study.optimize(func, n_trials=kwargs['n_trials'], n_jobs=-1)
    return study