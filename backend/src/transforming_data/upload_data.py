"""
Программа: Сохранение данных
Версия: 1.0
"""

import yaml
import pandas as pd

from .transform_data import pipeline_preprocessing


def load_data(config_path) -> None:

    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preproc = config['preprocessing']

    df_train, df_test, df_recommend_submission, \
        df_winner_submission, df_evaluate = pipeline_preprocessing(preproc)

    df_train.to_csv(preproc['train_data'], index_label='index')
    df_test.to_csv(preproc['test_data'], index_label='index')
    df_recommend_submission.to_csv(preproc['recommend_sub_path'], index_label='index')
    df_winner_submission.to_csv(preproc['winner_sub_path'], index_label='index')
    df_evaluate.to_csv(preproc['evaluate_data'], index_label='index')

