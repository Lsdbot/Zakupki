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


def merge_dataframes(column: str, *dfs) -> pd.DataFrame:
    return pd.merge(dfs, on=column)

def transform_ids(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].apply(lambda x: x.split('_')[1])

def replace_values(df: pd.DataFrame, pattern: dict) -> pd.DataFrame:
    return df.replace(pattern)

def fill_nulls(df: pd.DataFrame,
               column_nulls: str,
               column_values: str) -> pd.Series:
    return df[column_nulls].fillna(df[column_values])

def sort_dataframe(df: pd.DataFrame, *columns) -> pd.DataFrame:
    return df.sort_values(columns)

def lemmatize_text(row: dict, nlp, pos_to_include) -> list:
    doc = nlp(sum(value for value in row.values()))
    return list(_.lemma_ for _ in doc if _.pos_ in pos_to_include)

def process_rows(df: pd.DataFrame, nlp, pos_to_include: set,
                 columns: list) -> pd.Series:
    with mp.Pool() as pool:
        results = pool.map(lemmatize_text, df[columns].to_dict('records'),
                           nlp, pos_to_include)
    return results

def get_tokens(df: pd.DataFrame, text_columns: list) -> pd.Series:
    nlp = spacy.load("ru_core_news_sm")
    pos_to_include = {'ADJ', 'NOUN', 'PROPN'}

    return process_rows(df, nlp, pos_to_include, text_columns)

def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df.drop(columns=columns)

def vectorize_tfidf_matrix(df_column: pd.Series, n_components):
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

def pipeline_transform(df, **kwargs) -> pd.DataFrame:
    for column in kwargs['change_ids_columns']:
        df[column] = transform_ids(df, column)

    df = df.replace(kwargs['change_columns'])

    for column in kwargs['fillna_columns']:
        df[column] = fill_nulls(kwargs['fillna_columns'][column])

    df = df.sort_values(kwargs['sort_columns'])

    df['tokens'] = get_tokens(df, kwargs['text_columns'])
    df = df.drop(columns=kwargs['text_columns'])

    df['vectorized'] = list(vectorize_tfidf_matrix(df['tokens'], kwargs['n_components']))
    df= df.drop('tokens', axis=1)

    return df

def filter_data(df:pd.DataFrame, column: str, size: int) -> pd.DataFrame:
    true_false_seria = df.groupby(column).size() > size
    return df[df[column].isin(true_false_seria[true_false_seria].index)]

def split_data(df: pd.DataFrame, test_size,
               random_state, stratify_column) -> tuple:
    return train_test_split(df, test_size=test_size,
                            random_state=random_state,
                            stratify=df[stratify_column])

def create_recommend_submission(df: pd.DataFrame, column_groupby: str,
                                column_values, name) -> pd.DataFrame:
    return df.groupby(column_groupby)[column_values].apply(
        list).to_frame(name)

def create_winner_submission(df: pd.DataFrame, column: str,
                             name: str) -> pd.DataFrame:
    return df[column].to_frame(name)

def pipeline_split(df, **kwargs) -> tuple:
    df = filter_data(df, kwargs['filter_column'])

    tt_split = kwargs['train_test_split']
    df_train, df_test = train_test_split(df, test_size=tt_split['test_size'],
                                         random_state=tt_split['random_state'],
                                         stratify=tt_split['stratify'])

    rec_sub = kwargs['recommend_submission']
    df_recommend_submission = create_recommend_submission(df, rec_sub['groupby'],
                                                          rec_sub['values'],
                                                          rec_sub['name'])

    win_sub = kwargs['winner_submission']
    df_winner_submission = create_winner_submission(df, win_sub['column'], win_sub['name'])

    return df_train, df_test, df_recommend_submission, df_winner_submission


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

def pipeline_feature_engineering(df_train: pd.DataFrame, df_test: pd.DataFrame,
                                 **kwargs) -> tuple:

    df_train['month'] = extract_month(df_train, kwargs['date'])
    df_test['month'] = extract_month(df_test, kwargs['date'])

    df_train['reg_code'] = extract_reg_code(df_train, kwargs['region'],
                                            kwargs['okpd2'])
    df_test['reg_code'] = extract_reg_code(df_test, kwargs['region'],
                                           kwargs['okpd2'])

    purchase = kwargs['purhcase']
    df_train = extract_purchase_size(df_train, purchase['groupby'],
                                     purchase['values'], purchase['name'],
                                     purchase['on'], purchase['how'])
    df_test = extract_purchase_size(df_test, purchase['groupby'],
                                     purchase['values'], purchase['name'],
                                     purchase['on'], purchase['how'])

    flag = kwargs['flag']
    df_train, df_test = extract_flag(df_train, df_test, flag['train_columns'],
                                     flag['groupby'], flag['on'], flag['how'])

    uniq_okpd2 = kwargs['uniq_okpd2']
    df_train, df_test = extract_unique_okpd2(df_train, df_test, uniq_okpd2['groupby'],
                                             uniq_okpd2['values'], uniq_okpd2['on'],
                                             uniq_okpd2['name'])

    df_train = df_train.drop(columns=kwargs['drop_columns'])
    df_test = df_test.drop(columns=kwargs['drop_columns'])

    return df_train, df_test

