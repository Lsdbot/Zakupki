"""
Программа: Тренировка данных
Версия: 1.0
"""

import pandas as pd
import numpy as np

from catboost import CatBoostClassifier

import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def objective(trial: optuna.Trial, x: pd.DataFrame, y: pd.Series, **kwargs) -> np.ndarray:

    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [1000]),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0001, 100),
        'random_strength': trial.suggest_float('random_strength', 10, 50),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 100),
        'border_count': trial.suggest_categorical('border_count', [128]),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'random_state': kwargs['random_state']
    }

    cv_pred = np.empty(kwargs['N_FOLDS'])
    cv = StratifiedKFold(n_splits=kwargs['N_FOLDS'], shuffle=True, random_state=kwargs['random_state'])

    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        x_train_, x_val_ = x.iloc[train_idx], x.iloc[test_idx]
        y_train_, y_val_ = y.iloc[train_idx], y.iloc[test_idx]

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


def train_model(x_train, y_train, cat_features, params) -> CatBoostClassifier:

    ratio = y_train[y_train == 0].shape[0] / y_train[y_train == 1].shape[0]

    model = CatBoostClassifier(scale_pos_weight=ratio,
                               cat_features=cat_features,
                               **params)

    model.fit(x_train, y_train, verbose=0)

    return model


def find_optimal_params(x_train, y_train, **kwargs) -> optuna.Study:

    func = lambda trial: objective(trial, x_train, y_train,
                                   N_FOLDS=kwargs['N_FOLDS'],
                                   random_state=kwargs['random_state'],
                                   cat_features=kwargs['cat_features'])

    study = optuna.create_study(direction="maximize")
    study.optimize(func, show_progress_bar=True, n_trials=kwargs['n_trials'], n_jobs=6)

    return study
