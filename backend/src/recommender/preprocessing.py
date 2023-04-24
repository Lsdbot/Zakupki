"""
Программа: Предобработка данных
Версия: 1.0
"""

from pandas import DataFrame


def get_supplier_data(df_train: DataFrame, df_test: DataFrame, sup: int, supplier_purchases: list,
                      **kwargs) -> tuple:
    """
    Получает данные поставщика и фильтрует их на основе уникальных reg_code поставщиков.
    Удаляет ненужные для системы рекомендаций столбцы и дубликаты.
    Удаляет закупки, которые есть и test, и в train.

    :param df_train: обучающая выборка
    :param df_test: тестовая выборка
    :param sup: код поставщика
    :param supplier_purchases: закупки поставщика
    :param kwargs: дополнительные аргументы - filter_column, drop_columns, index_column

    :return: кортеж из фильтрованных train и test данных поставщика
    """
    unique_reg_okpd = df_train[df_train[kwargs['sup_column']] == sup][kwargs['filter_column']].unique()

    # фильтруем train на основе уникальных reg_code поставщиков
    df_sup_train = df_train[df_train[kwargs['filter_column']].isin(unique_reg_okpd)]
    df_sup_test = df_test[df_test[kwargs['filter_column']].isin(unique_reg_okpd)]

    if df_sup_test.empty:
        df_sup_test = df_test

    # удаляем ненужные для системы рекомендаций столбцы и дубликаты
    df_sup_train = df_sup_train.drop(columns=kwargs['drop_columns']).drop_duplicates()
    df_sup_test = df_sup_test.drop(columns=kwargs['drop_columns']).drop_duplicates()

    df_sup_train = df_sup_train.set_index(kwargs['index_column'])
    df_sup_test = df_sup_test.set_index(kwargs['index_column'])

    # удаляем закупки, которые есть и test, и в train
    df_sup_train = df_sup_train.drop(set(supplier_purchases).intersection(df_sup_train.index))
    df_sup_test = df_sup_test[~df_sup_test.index.isin(df_sup_train.index)]

    return df_sup_train, df_sup_test


