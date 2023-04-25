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


def pipeline_transform(preproc: dict, df_sup: pd.DataFrame, df_pur: pd.DataFrame) -> pd.DataFrame:
    """
    Обрабатывает данные, объединяет таблицы, преобразует и заполняет значения, векторизует текст и возвращает новый DataFrame.

    Args:
        preproc (dict): Словарь параметров предобработки данных.
        df_sup (pd.DataFrame): DataFrame, содержащий сырые данные о поставках.
        df_pur (pd.DataFrame): DataFrame, содержащий сырые данные о закупках.

    Returns:
        pd.DataFrame: DataFrame с обработанными данными.

    """

    # Объединение таблиц по столбцу merge_column
    df = pd.merge(df_sup, df_pur, on=preproc['merge_column'])

    # Преобразование столбцов с идентификаторами
    for column in preproc['change_ids_columns']:
        df[column] = transform_ids(df, column)

    # Замена значений в соответствии с change_columns
    df = df.replace(preproc['change_columns'])

    # Заполнение пропущенных значений в столбцах fillna_columns
    for column in preproc['fillna_columns']:
        df[column] = df[column].fillna(df[preproc['fillna_columns'][column]])

    # Сортировка по столбцам sort_columns
    df = df.sort_values(preproc['sort_columns'])

    # Получение токенов из текстовых столбцов и удаление их
    df['tokens'] = get_tokens(df)
    df = df.drop(columns=preproc['text_columns'])

    # Векторизация текстовых столбцов с помощью TF-IDF и сохранение векторов в новый столбец
    df[preproc['vector']] = list(vectorize_tfidf_matrix(df['tokens'], preproc['n_components']))
    df = df.drop('tokens', axis=1)

    return df


def filter_data(df: pd.DataFrame, column: str, size: int) -> pd.DataFrame:
    """
    Отбирает данные DataFrame по значению в столбце, которое встречается чаще заданного количества раз.

    Args:
        df (pd.DataFrame): DataFrame для фильтрации.
        column (str): Название столбца для фильтрации.
        size (int): Минимальное количество вхождений значения в столбце для сохранения в результирующем DataFrame.

    Returns:
        pd.DataFrame: DataFrame с отфильтрованными данными.

    """

    # Получение серии с булевыми значениями в зависимости от частоты вхождений каждого значения
    true_false_seria = df.groupby(column).size() > size

    # Отбор строк DataFrame, значение в столбце которых встречается чаще size раз
    return df[df[column].isin(true_false_seria[true_false_seria].index)]



def pipeline_split(df: pd.DataFrame, preproc: dict) -> tuple:
    """
    Разбивает DataFrame на обучающую и тестовую выборки и отбирает данные по значению в столбце, которое встречается
    чаще заданного количества раз.

    Args:
        df (pd.DataFrame): DataFrame для разбиения.
        preproc (dict): Настройки предобработки данных, содержащие параметры разбиения, фильтрации и т.д.

    Returns:
        tuple: Кортеж, содержащий два DataFrame - обучающую и тестовую выборки.

    """

    # Фильтрация DataFrame по частоте вхождения значений в столбец
    df = filter_data(df, preproc['filter_column'], preproc['size'])

    # Разбиение DataFrame на обучающую и тестовую выборки
    tt_split = preproc['train_test_split']
    df_train, df_test = train_test_split(df, test_size=tt_split['test_size'],
                                         random_state=tt_split['random_state'],
                                         stratify=df[tt_split['stratify']])

    return df_train, df_test


def extract_month(df, column) -> pd.Series:
    return df[column].apply(lambda x: int(x.split('-')[1]))


def extract_reg_code(df, column_region, column_okpd2) -> pd.Series:
    return df[column_region].astype('str') + '_' + df[column_okpd2].astype('str')


def extract_purchase_size(df, group, values, name, on_col,) -> pd.DataFrame:
    return df.merge(df.groupby(group)[values].size().to_frame(name),
                    on=on_col, how='outer')


def extract_flag(df_train: pd.DataFrame, df_test: pd.DataFrame, train_columns: list,
                 group: list, on_col: list) -> tuple:
    df_train['flag_won'] = df_train[df_train.is_winner == 1].groupby(
        group).cumcount().apply(lambda x: 1 if x != 0 else 0)

    df_test = df_test.merge(df_train[train_columns].groupby(group).tail(1),
                            on=on_col, how='left').fillna(0)

    return df_train.fillna(0), df_test


def extract_unique_okpd2(df_train: pd.DataFrame, df_test: pd.DataFrame, group: str,
                         values: str, on_col: str, name: str) -> tuple:
    df_train = df_train.merge(df_train.groupby(group)[values].nunique().to_frame(name),
                              how='outer', on=on_col)

    df_test = df_test.merge(df_train[[group, name]].groupby(group).tail(1),
                            on=on_col, how='left').fillna(1)

    return df_train, df_test


def pipeline_feature_engineering(preproc: dict, df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    """
    Функция для преобразования данных тренировочного и тестового наборов с помощью инженерии признаков.

    Аргументы:
    - preproc (dict): Словарь с параметрами для инженерии признаков.
    - df_train (pd.DataFrame): Данные для тренировки.
    - df_test (pd.DataFrame): Данные для тестирования.

    Возвращаемое значение:
    - tuple: Кортеж, содержащий обработанные данные для тренировки и тестирования.
    """

    # Извлечение месяца из даты
    df_train['month'] = extract_month(df_train, preproc['date'])
    df_test['month'] = extract_month(df_test, preproc['date'])

    # Извлечение кода региона и ОКПД2
    df_train['reg_code'] = extract_reg_code(df_train, preproc['region'], preproc['okpd2'])
    df_test['reg_code'] = extract_reg_code(df_test, preproc['region'], preproc['okpd2'])

    # Извлечение размера покупки
    purchase_s = preproc['purchase_size']
    df_train = extract_purchase_size(df_train, purchase_s['group'], purchase_s['values'],
                                     purchase_s['name'], purchase_s['on_col'])
    df_test = extract_purchase_size(df_test, purchase_s['group'], purchase_s['values'],
                                    purchase_s['name'], purchase_s['on_col'])

    # Извлечение флага
    flag = preproc['flag']
    df_train, df_test = extract_flag(df_train, df_test, flag['train_columns'],
                                     flag['group'], flag['on_col'])

    # Извлечение кол-ва уникальных кодов ОКПД2
    uniq_okpd2 = preproc['uniq_okpd2']
    df_train, df_test = extract_unique_okpd2(df_train, df_test, uniq_okpd2['group'], uniq_okpd2['values'],
                                             uniq_okpd2['on_col'], uniq_okpd2['name'])

    # Удаление ненужных столбцов
    df_train = df_train.drop(columns=preproc['drop_columns'])
    df_test = df_test.drop(columns=preproc['drop_columns'])

    return df_train, df_test


def pipeline_preprocessing(preproc: dict) -> tuple:
    """
    Обработка данных для последующей моделирования.

    Args:
    preproc (Dict[str, any]): Словарь параметров для обработки данных.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Кортеж из пяти pandas DataFrame.
        df_train - обучающий набор данных;
        df_test - тестовый набор данных;
        df_recommend_submission - рекомендуемый набор данных;
        df_winner_submission - набор данных победителя;
        df_evaluate - оценочный набор данных.
    """

    # Получение параметров из preproc
    rec_sub = preproc['recommend_submission']
    win_sub = preproc['winner_submission']

    # Получение данных из источников
    df_sup = get_data(preproc['supplier_path'])
    df_pur = get_data(preproc['purchases_path'])

    # Применение трансформации
    df = pipeline_transform(preproc, df_sup, df_pur)

    # Разделение данных на тренировочный и тестовый наборы
    df_train, df_test = pipeline_split(df, preproc)

    # Выделение данных для оценки модели
    df_evaluate = df_test.copy()

    # Применение feature engineering
    df_train, df_test = pipeline_feature_engineering(preproc, df_train, df_test)

    # Группировка тестовых данных для рекомендации
    df_recommend_submission = df_test.groupby(rec_sub['group'])[rec_sub['values']].apply(list).to_frame(rec_sub['name'])

    # Выбор победителя
    df_winner_submission = df_test[win_sub['column']].to_frame(win_sub['name'])

    return df_train, df_test, df_recommend_submission, df_winner_submission, df_evaluate
