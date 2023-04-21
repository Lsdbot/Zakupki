"""
Программа: Получение предсказания на основе обученной модели
Версия: 1.0
"""

import pandas as pd

from ..transforming_data.transform_data import extract_month, extract_reg_code, extract_purchase_size


def transform_vector(df, **kwargs):
    for i in range(kwargs['n_components']):
        df[str(i)] = df[kwargs['vector']].apply(lambda x: x[i])

    return df


def extract_flag(train_data: pd.DataFrame, df_test: pd.DataFrame, flag_columns: list,
                 group: list, on: list) -> pd.DataFrame:
    return df_test.merge(train_data[flag_columns].groupby(group).tail(1),
                         on=on, how='left').fillna(0)


def extract_unique_okpd2(train_data: pd.DataFrame, df_test: pd.DataFrame,
                         group: str, on: str, name: str) -> pd.DataFrame:
    return df_test.merge(train_data[[group, name]].groupby(group).tail(1),
                         on=on, how='left').fillna(1)


def generate_features(train_data, df_test, preproc) -> pd.DataFrame:

    df_test['month'] = extract_month(df_test, preproc['date'])

    df_test['reg_code'] = extract_reg_code(df_test, preproc['region'],
                                           preproc['okpd2'])

    purchase = preproc['purchase_size']
    df_test = extract_purchase_size(df_test, purchase['group'],
                                    purchase['values'], purchase['name'],
                                    purchase['on_col'])

    flag = preproc['flag']
    df_test = extract_flag(train_data, df_test, flag['train_columns'],
                           flag['group'], flag['on_col'])

    uniq_okpd2 = preproc['uniq_okpd2']
    df_test = extract_unique_okpd2(train_data, df_test, uniq_okpd2['group'],
                                   uniq_okpd2['on_col'], uniq_okpd2['name'])

    df_test = df_test.drop(columns=preproc['drop_columns'])

    return df_test


def get_supplier_data(df_train, df_test, sup, supplier_purchases, **kwargs):
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

    return df_sup_test
