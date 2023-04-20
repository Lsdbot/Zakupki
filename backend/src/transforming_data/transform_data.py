"""
Программа: Преобразование данных
Версия: 1.0
"""

import pandas as pd

import multiprocessing as mp
import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from .get_data import get_data


nlp = spacy.load("ru_core_news_sm")

def merge_dataframes(column: str, *dfs) -> pd.DataFrame:
    return pd.merge(dfs, on=column)


def transform_ids(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].apply(lambda x: x.split('_')[1])


def lemmatize_text(row: dict) -> list:
    doc = nlp(row['purchase_name'] + row['okpd2_names'] + row['item_descriptions'])
    return list(_.lemma_ for _ in doc if _.pos_ in {'ADJ', 'NOUN', 'PROPN'})


def get_tokens(df: pd.DataFrame) -> pd.Series:
    with mp.Pool() as pool:
        results = pool.map(lemmatize_text, df.to_dict('records'))
    return results


def vectorize_tfidf_matrix(df_column: pd.Series, n_components: int):
    """
    This function takes a pandas DataFrame column, flattens the data and applies
    a TF-IDF vectorizer to convert the text data into a sparse matrix. It then
    applies TruncatedSVD to reduce the dimensionality of the matrix to 100 components.
    The resulting matrix is then yielded row by row.

    Args:
        df_column (Iterable): A column of text data.

    Yields:
        Iterable: A row of the transformed matrix.
    """
    flattened_list = [' '.join(words) for words in df_column]

    # создаем объект TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # создаем разреженную матрицу TF-IDF
    tfidf_matrix = tfidf_vectorizer.fit_transform(flattened_list)

    # примененяем TruncatedSVD к матрице, оставляя n компонент
    svd = TruncatedSVD(n_components=n_components)
    tfidf_svd = svd.fit_transform(tfidf_matrix)

    for row in tfidf_svd:
        yield row


def pipeline_transform(preproc, df_sup, df_pur) -> pd.DataFrame:

    df = pd.merge(df_sup, df_pur, on=preproc['merge_column'])

    for column in preproc['change_ids_columns']:
        df[column] = transform_ids(df, column)

    df = df.replace(preproc['change_columns'])

    for column in preproc['fillna_columns']:
        df[column] = df[column].fillna(df[preproc['fillna_columns'][column]])

    df = df.sort_values(preproc['sort_columns'])

    df['tokens'] = get_tokens(df)
    df = df.drop(columns=preproc['text_columns'])

    df['vectorized'] = list(vectorize_tfidf_matrix(df['tokens'], preproc['n_components']))
    df = df.drop('tokens', axis=1)

    return df


def filter_data(df: pd.DataFrame, column: str, size: int) -> pd.DataFrame:
    true_false_seria = df.groupby(column).size() > size
    return df[df[column].isin(true_false_seria[true_false_seria].index)]


def pipeline_split(df, preproc) -> tuple:

    df = filter_data(df, preproc['filter_column'], preproc['size'])

    tt_split = preproc['train_test_split']
    df_train, df_test = train_test_split(df, test_size=tt_split['test_size'],
                                         random_state=tt_split['random_state'],
                                         stratify=tt_split['stratify'])

    return df_train, df_test


def extract_month(df, column) -> pd.Series:
    return df[column].apply(lambda x: int(x.split('-')[1]))


def extract_reg_code(df, column_region, column_okpd2) -> pd.Series:
    return df[column_region].astype('str') + '_' + df[column_okpd2].astype('str')


def extract_purchase_size(df, groupby, values, name, on, how) -> pd.DataFrame:
    return df.merge(df.groupby(groupby)[values].size().to_frame(name),
                    on=on, how=how)


def extract_flag(df_train: pd.DataFrame, df_test: pd.DataFrame, train_columns: list,
                 groupby: list, on: list, how: str) -> tuple:
    df_train['flag_won'] = df_train[df_train.is_winner == 1].groupby(
        groupby).cumcount().apply(lambda x: 1 if x != 0 else 0)

    df_test = df_test.merge(df_train[train_columns].groupby(groupby).tail(1),
                            on=on, how=how).fillna(0)

    return df_train.fillna(0), df_test


def extract_unique_okpd2(df_train: pd.DataFrame, df_test: pd.DataFrame, groupby: str,
                         values: str, on: str, name: str) -> tuple:
    df_train = df_train.merge(df_train.groupby(groupby)[values].nunique().to_frame(name),
                              how='outer', on=on)

    df_test = df_test.merge(df_train[[groupby, name]].groupby(groupby).tail(1),
                            on=on, how='left').fillna(1)

    return df_train, df_test


def pipeline_feature_engineering(preproc, df_train, df_test) -> tuple:

    df_train['month'] = extract_month(df_train, preproc['date'])
    df_test['month'] = extract_month(df_test, preproc['date'])

    df_train['reg_code'] = extract_reg_code(df_train, preproc['region'],
                                            preproc['okpd2'])
    df_test['reg_code'] = extract_reg_code(df_test, preproc['region'],
                                           preproc['okpd2'])

    purchase = preproc['purchase']
    df_train = extract_purchase_size(df_train, purchase['group'],
                                     purchase['values'], purchase['name'],
                                     purchase['on'], purchase['how'])
    df_test = extract_purchase_size(df_test, purchase['group'],
                                    purchase['values'], purchase['name'],
                                    purchase['on'], purchase['how'])

    flag = preproc['flag']
    df_train, df_test = extract_flag(df_train, df_test, flag['train_columns'],
                                     flag['group'], flag['on'], flag['how'])

    uniq_okpd2 = preproc['uniq_okpd2']
    df_train, df_test = extract_unique_okpd2(df_train, df_test, uniq_okpd2['group'],
                                             uniq_okpd2['values'], uniq_okpd2['on'],
                                             uniq_okpd2['name'])

    df_train = df_train.drop(columns=preproc['drop_columns'])
    df_test = df_test.drop(columns=preproc['drop_columns'])

    return df_train, df_test


def pipeline_preprocessing(preproc):

    rec_sub = preproc['recommend_submission']
    win_sub = preproc['winner_submission']

    df_sup = get_data(preproc['supplier_path'])
    df_pur = get_data(preproc['purchases_path'])

    df = pipeline_transform(preproc, df_sup, df_pur)

    df_train, df_test = pipeline_split(df, preproc)

    df_evaluate = df_test.copy()

    df_train, df_test = pipeline_feature_engineering(preproc, df_train, df_test)

    df_recommend_submission = df.groupby(rec_sub['group'])[rec_sub['values']].apply(
        list).to_frame(rec_sub['name'])

    df_winner_submission = df[win_sub['column']].to_frame(win_sub['name'])

    return df_train, df_test, df_recommend_submission, df_winner_submission, df_evaluate
