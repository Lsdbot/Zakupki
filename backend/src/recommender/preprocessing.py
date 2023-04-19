"""
Программа: Предобработка данных
Версия: 1.0
"""


def get_supplier_data(df_train, df_submission, sup, kwargs):
    unique_reg_okpd = df_train[df_train[kwargs['suppplier_column']] == sup][kwargs['filter_column']].unique()

    # фильтруем train на основе уникальных reg_code поставщиков
    df_sup_train = df_train[df_train['filter_column'].isin(unique_reg_okpd)]

    # удаляем ненужные для системы рекомендаций столбцы и дубликаты
    df_sup_train = df_sup_train.drop(columns=kwargs['drop_columns']).drop_duplicates()

    df_sup_train = df_sup_train.set_index(kwargs['index_column'])

    # удаляем закупки, которые есть и test, и в train
    df_sup_train = df_sup_train.drop(set(df_submission[sup]).intersection(df_sup_train.index))

    return df_sup_train
