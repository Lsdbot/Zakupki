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

from get_data import get_data



def merge_dataframes(column: str, *dfs) -> pd.DataFrame:
    return pd.merge(dfs, on=column)

def transform_ids(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].apply(lambda x: x.split('_')[1])

def fill_nulls(df: pd.DataFrame,
               column_nulls: str,
               column_values: str) -> pd.Series:
    return df[column_nulls].fillna(df[column_values])

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

def pipeline_transform(preproc, df_sup, df_pur) -> pd.DataFrame:

    df = pd.merge(df_sup, df_pur, on=preproc['merge_column'])

    for column in preproc['change_ids_columns']:
        df[column] = transform_ids(df, column)

    df = df.replace(preproc['change_columns'])

    for column in preproc['fillna_columns']:
        df[column] = fill_nulls(preproc['fillna_columns'][column])

    df = df.sort_values(preproc['sort_columns'])

    df['tokens'] = get_tokens(df, preproc['text_columns'])
    df = df.drop(columns=preproc['text_columns'])

    df['vectorized'] = list(vectorize_tfidf_matrix(df['tokens'], preproc['n_components']))
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

def pipeline_split(preproc, df) -> tuple:

    df = filter_data(df, preproc['filter_column'])

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

    purchase = preproc['purhcase']
    df_train = extract_purchase_size(df_train, purchase['groupby'],
                                     purchase['values'], purchase['name'],
                                     purchase['on'], purchase['how'])
    df_test = extract_purchase_size(df_test, purchase['groupby'],
                                     purchase['values'], purchase['name'],
                                     purchase['on'], purchase['how'])

    flag = preproc['flag']
    df_train, df_test = extract_flag(df_train, df_test, flag['train_columns'],
                                     flag['groupby'], flag['on'], flag['how'])

    uniq_okpd2 = preproc['uniq_okpd2']
    df_train, df_test = extract_unique_okpd2(df_train, df_test, uniq_okpd2['groupby'],
                                             uniq_okpd2['values'], uniq_okpd2['on'],
                                             uniq_okpd2['name'])

    df_train = df_train.drop(columns=preproc['drop_columns'])
    df_test = df_test.drop(columns=preproc['drop_columns'])

    return df_train, df_test

def pipeline_preprocessing(preproc):

    rec_sub = preproc['recommend_submission']
    win_sub = preproc['winner_submission']

    df_sup = get_data(preproc['supplier_small_path'])
    df_pur = get_data(preproc['purchase_small_path'])

    df = pipeline_transform(df_sup, df_pur)

    df_train, df_test = pipeline_split(preproc, df)

    df_evaluate = df_test.copy()

    df_train, df_test = pipeline_feature_engineering(preproc, df_train, df_test)

    df_recommend_submission = create_recommend_submission(df_test, rec_sub['groupby'],
                                                          rec_sub['values'],
                                                          rec_sub['name'])

    df_winner_submission = create_winner_submission(df_test, win_sub['column'],
                                                    win_sub['name'])

    return df_train, df_test, df_recommend_submission, df_winner_submission, df_evaluate