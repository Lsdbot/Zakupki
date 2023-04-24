"""
Программа: Сохранение данных
Версия: 1.0
"""

import yaml

from .transform_data import pipeline_preprocessing



def load_data(config_path: str) -> None:
    """
    Загрузка данных из файла конфигурации и сохранение обработанных данных в CSV-файлы.

    Args:
    config_path (str): Путь к файлу конфигурации.

    Returns:
    None
    """

    # Загрузка конфигурации
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Получение параметров для предобработки данных
    preproc = config['preprocessing']

    # Обработка данных с использованием pipeline_preprocessing
    df_train, df_test, df_recommend_submission, \
        df_winner_submission, df_evaluate = pipeline_preprocessing(preproc)

    # Сохранение данных в CSV-файлы
    df_train.to_csv(preproc['train_data'], index_label='index')
    df_test.to_csv(preproc['test_data'], index_label='index')
    df_recommend_submission.to_csv(preproc['recommend_sub_path'], index_label='index')
    df_winner_submission.to_csv(preproc['winner_sub_path'], index_label='index')
    df_evaluate.to_csv(preproc['evaluate_data'], index_label='index')

