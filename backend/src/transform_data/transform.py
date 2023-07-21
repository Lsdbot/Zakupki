"""
Программа: Преобразование данных
Версия: 1.0
"""

import pandas as pd

import multiprocessing as mp
import ru_core_news_sm
import yaml

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from ..data.get_data import get_data
from ..upload_data.upload import load_data


nlp = ru_core_news_sm.load()


def transform_ids(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Извлекает числовые идентификаторы из колонки `column` в датафрейме `df` и возвращает их в виде pandas.Series.

    Аргументы:
        df (pd.DataFrame): Исходный датафрейм.
        column (str): Название колонки, из которой нужно извлечь числовые идентификаторы.

    Возвращает:
        pd.Series: Pandas.Series, содержащий числовые идентификаторы, извлеченные из указанной колонки датафрейма.

    """
    return df[column].apply(lambda x: x.split('_')[1])


def transform_vector(df: pd.DataFrame, n_components: int, vector_column: str):
    """
    Добавляет столбцы с разложением вектора на n_components компонент.

    Аргументы:
        df (pd.DataFrame): Исходный DataFrame.
        n_components (int): Количество компонент разложения.
        vector (str): Название столбца с вектором.

    Возвращает:
        pd.DataFrame: DataFrame с добавленными столбцами.
    """
    for i in range(n_components):
        df[str(i)] = df[vector_column].apply(lambda x: x[i])

    return df


def lemmatize_text(row: dict) -> list:
    """
    Производит лемматизацию текста в колонках датафрейма `row` с помощью библиотеки spaCy.

    Аргументы:
        row (dict): Словарь, содержащий колонки датафрейма с текстовыми данными.

    Возвращает:
        list: Список лемм слов в колонках датафрейма.

    """
    doc = nlp(row['purchase_name'] + row['okpd2_names'] + row['item_descriptions'])
    return list(_.lemma_ for _ in doc if _.pos_ in {'ADJ', 'NOUN', 'PROPN'})


def get_tokens(df: pd.DataFrame) -> pd.Series:
    """
    Производит лемматизацию текста в колонках датафрейма `df` и возвращает леммы в виде pandas.Series.

    Аргументы:
        df (pd.DataFrame): Исходный датафрейм.

    Возвращает:
        pd.Series: Pandas.Series, содержащий леммы текста в колонках датафрейма.

    """
    with mp.Pool() as pool:
        results = pool.map(lemmatize_text, df.to_dict('records'))
    return results


def vectorize_tfidf_matrix(df_column: pd.Series, n_components: int):
    """
    Принимает столбец pandas DataFrame, выравнивает данные и применяет
    векторизатор TF-IDF для преобразования текстовых данных в разреженную матрицу.
    Затем она применяет метод TruncatedSVD для уменьшения размерности матрицы.
    Результатирующая матрица затем возвращается строка за строкой.

    Аргументы:
        df_column (Iterable): Столбец текстовых данных.
        n_components (int): Количество компонентов векторизованных текстов.

    Возвращает:
        Iterable: Строка преобразованной матрицы.
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


def transform_data(df_sup: pd.DataFrame, df_pur: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Обрабатывает данные, объединяет таблицы, преобразует и заполняет значения, векторизует текст и
    возвращает новый DataFrame.

    Аргументы:
        df_sup (pd.DataFrame): DataFrame, содержащий сырые данные о поставках.
        df_pur (pd.DataFrame): DataFrame, содержащий сырые данные о закупках.

    Возвращает:
        pd.DataFrame: DataFrame с обработанными данными.

    """

    # Объединение таблиц по столбцу merge_column
    df = pd.merge(df_sup, df_pur, on=kwargs['merge_column'])

    # Выделение сэмпла данных
    df = df.sample(frac=0.2, random_state=kwargs['train_test_split']['random_state'])

    # Преобразование столбцов с идентификаторами
    for column in kwargs['change_ids_columns']:
        df[column] = transform_ids(df, column)

    # Замена значений в соответствии с change_columns
    df = df.replace(kwargs['change_columns'])

    # Заполнение пропущенных значений в столбцах fillna_columns
    for column in kwargs['fillna_columns']:
        df[column] = df[column].fillna(df[kwargs['fillna_columns'][column]])

    # Сортировка по столбцам sort_columns
    df = df.sort_values(kwargs['sort_columns'])

    # Получение токенов из текстовых столбцов и удаление их
    df['tokens'] = get_tokens(df)
    df = df.drop(columns=kwargs['text_columns'])

    # Векторизация текстовых столбцов с помощью TF-IDF и сохранение векторов в новый столбец
    df[kwargs['vector']] = list(vectorize_tfidf_matrix(df['tokens'], kwargs['n_components']))
    df = df.drop('tokens', axis=1)

    return df


def filter_data(df: pd.DataFrame, column: str, size: int) -> pd.DataFrame:
    """
    Отбирает данные DataFrame по значению в столбце, которое встречается чаще заданного количества раз.

    Аргументы:
        df (pd.DataFrame): DataFrame для фильтрации.
        column (str): Название столбца для фильтрации.
        size (int): Минимальное количество вхождений значения в столбце для сохранения в результирующем DataFrame.

    Возвращает:
        pd.DataFrame: DataFrame с отфильтрованными данными.

    """

    # Получение серии с булевыми значениями в зависимости от частоты вхождений каждого значения
    true_false_seria = df.groupby(column).size() > size

    # Отбор строк DataFrame, значение в столбце которых встречается чаще size раз
    return df[df[column].isin(true_false_seria[true_false_seria].index)]


def split_data(df: pd.DataFrame, **kwargs) -> tuple:
    """
    Разбивает DataFrame на обучающую и тестовую выборки и отбирает данные по значению в столбце, которое встречается
    чаще заданного количества раз.

    Аргументы:
        df (pd.DataFrame): DataFrame для разбиения.
        kwargs:
            filter_column (str): Имя столбца для фильтрации по частоте вхождения значений.
            size (int): Количество вхождений значений в столбце для отбора данных.
            train_test_split (dict): Словарь параметров разбиения на обучающую и тестовую выборки.
                test_size (float): Доля тестовой выборки.
                random_state (int): Случайное состояние генератора псевдослучайных чисел для разбиения.
                stratify (str or array-like): Столбец или массив для стратификации выборки.
    Возвращает:
        tuple: Кортеж, содержащий два DataFrame - обучающую и тестовую выборки.

    """

    # Фильтрация DataFrame по частоте вхождения значений в столбец
    df = filter_data(df, kwargs['filter_column'], kwargs['size'])

    # Разбиение DataFrame на обучающую и тестовую выборки
    tt_split = kwargs['train_test_split']
    df_train, df_test = train_test_split(df, test_size=tt_split['test_size'],
                                         random_state=tt_split['random_state'],
                                         stratify=df[tt_split['stratify']])

    return df_train, df_test


def extract_month(df, column) -> pd.Series:
    """
    Извлекает месяц из столбца даты в данном DataFrame.

    Аргументы:
        df (pd.DataFrame): DataFrame, содержащий столбец даты для извлечения.
        column (str): Название столбца даты для извлечения.

    Возвращает:
        pd.Series: Серия, содержащая извлеченные значения месяца из столбца даты.
    """
    return df[column].apply(lambda x: int(x.split('-')[1]))


def extract_reg_code(df, column_region, column_okpd2) -> pd.Series:
    """
    Извлекает код региона из двух столбцов данного DataFrame.

    Аргументы:
        df (pd.DataFrame): DataFrame, содержащий столбцы для извлечения.
        column_region (str): Название столбца с регионами для извлечения.
        column_okpd2 (str): Название столбца с кодами товарной номенклатуры для извлечения.

    Возвращает:
        pd.Series: Серия, содержащая коды региона, извлеченные из двух столбцов.
    """
    return df[column_region].astype('str') + '_' + df[column_okpd2].astype('str')


def extract_purchase_size(df, group, values, name, on_col) -> pd.DataFrame:
    """
    Извлекает размер покупок для каждой группы, заданной по определенному столбцу,
    и объединяет результаты с данным DataFrame.

    Аргументы:
        df (pd.DataFrame): DataFrame, содержащий столбец для группировки.
        group (str): Название столбца для группировки.
        values (str): Название столбца для подсчета размера покупок.
        name (str): Название столбца для объединения результатов.
        on_col (str): Название столбца для объединения.

    Возвращает:
        pd.DataFrame: DataFrame с размерами покупок для каждой группы, объединенный с данным DataFrame.
    """
    return df.merge(df.groupby(group)[values].size().to_frame(name),
                    on=on_col, how='outer')


def extract_flag(df_train: pd.DataFrame, df_test: pd.DataFrame, train_columns: list,
                 group: list, on_col: list) -> tuple:
    """
    Извлекает флаги для каждой покупки, показывающие, была ли эта покупка выигрышной для соответствующей группы.
    Объединяет результаты с данным DataFrame.

    Аргументы:
        df_train (pd.DataFrame): Обучающий DataFrame.
        df_test (pd.DataFrame): Тестовый DataFrame.
        train_columns (list): Список столбцов для извлечения.
        group (list): Список столбцов для группировки.
        on_col (list): Список столбцов для объединения.

    Возвращает:
        tuple: Кортеж из двух DataFrame, содержащий извлеченные флаги
    """
    df_train['flag_won'] = df_train[df_train.is_winner == 1].groupby(
        group).cumcount().apply(lambda x: 1 if x != 0 else 0)

    df_test = df_test.merge(df_train[train_columns].groupby(group).tail(1),
                            on=on_col, how='left').fillna(0)

    return df_train.fillna(0), df_test


def extract_unique_okpd2(df_train: pd.DataFrame, df_test: pd.DataFrame, group: str,
                         values: str, on_col: str, name: str) -> tuple:
    """
    Извлекает количество уникальных значений в столбце values для каждой группы из столбца group
    и добавляет столбец с результатом под именем name в обеих таблицах.

    Аргументы:
        df_train (pd.DataFrame): исходная таблица с данными для обучения модели.
        df_test (pd.DataFrame): таблица с данными для тестирования модели.
        group (str): название столбца, содержащего группы для агрегации.
        values (str): название столбца, содержащего значения, для которых необходимо определить уникальные значения.
        on_col (str): название столбца, используемого для объединения таблиц.
        name (str): название нового столбца, содержащего количество уникальных значений для каждой группы.

    Возвращает:
        tuple: Обновленную таблицу df_train и обновленную таблицу df_test.
    """
    df_train = df_train.merge(df_train.groupby(group)[values].nunique().to_frame(name),
                              how='outer', on=on_col)

    df_test = df_test.merge(df_train[[group, name]].groupby(group).tail(1),
                            on=on_col, how='left').fillna(1)

    return df_train, df_test


def generate_features(df_train: pd.DataFrame, df_test: pd.DataFrame, **kwargs) -> tuple:
    """
    Функция для преобразования данных тренировочного и тестового наборов с помощью инженерии признаков.

    Аргументы:
        df_train (pd.DataFrame): Данные для тренировки.
        df_test (pd.DataFrame): Данные для тестирования.

    Возвращает:
        tuple: Кортеж, содержащий обработанные данные для тренировки и тестирования.
    """

    # Извлечение месяца из даты
    df_train['month'] = extract_month(df_train, kwargs['date'])
    df_test['month'] = extract_month(df_test, kwargs['date'])

    # Извлечение кода региона и ОКПД2
    df_train['reg_code'] = extract_reg_code(df_train, kwargs['region'], kwargs['okpd2'])
    df_test['reg_code'] = extract_reg_code(df_test, kwargs['region'], kwargs['okpd2'])

    # Извлечение размера покупки
    purchase_s = kwargs['purchase_size']
    df_train = extract_purchase_size(df_train, purchase_s['group'], purchase_s['values'],
                                     purchase_s['name'], purchase_s['on_col'])
    df_test = extract_purchase_size(df_test, purchase_s['group'], purchase_s['values'],
                                    purchase_s['name'], purchase_s['on_col'])

    # Извлечение флага
    flag = kwargs['flag']
    df_train, df_test = extract_flag(df_train, df_test, flag['train_columns'],
                                     flag['group'], flag['on_col'])

    # Извлечение кол-ва уникальных кодов ОКПД2
    uniq_okpd2 = kwargs['uniq_okpd2']
    df_train, df_test = extract_unique_okpd2(df_train, df_test, uniq_okpd2['group'], uniq_okpd2['values'],
                                             uniq_okpd2['on_col'], uniq_okpd2['name'])

    # Удаление ненужных столбцов
    df_train = df_train.drop(columns=kwargs['drop_columns'])
    df_test = df_test.drop(columns=kwargs['drop_columns'])

    return df_train, df_test


def get_supplier_data(df_train: pd.DataFrame, df_test: pd.DataFrame, sup: int, supplier_purchases: list,
                      **kwargs) -> tuple:
    """
    Получает данные поставщика и фильтрует их на основе уникальных reg_code поставщиков.
    Удаляет ненужные для системы рекомендаций столбцы и дубликаты.
    Удаляет закупки, которые есть и test, и в train.

    Аргументы:
        df_train: обучающая выборка
        df_test: тестовая выборка
        sup: код поставщика
        supplier_purchases: закупки поставщика
        kwargs: дополнительные аргументы - filter_column, drop_columns, index_column

    Возвращет:
        tuple: Кортеж из фильтрованных train и test данных поставщика
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


def pipeline_preprocessing(config_path: str) -> None:
    """
    Обработка данных для последующей моделирования.

    Аргументы:
        config_path (str): Путь до конфигурационного файла.
    """
    # загрузка конфигурации модели
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Получение параметров для предобработки
    preproc = config['preproccesing']

    # Получение параметров из preproс
    rec_sub = preproc['recommend_submission']
    df_sup = get_data(preproc['supplier_path'])
    df_pur = get_data(preproc['purchases_path'])

    # Применение трансформации
    df = transform_data(df_sup, df_pur, **preproc)

    # Разделение данных на тренировочный и тестовый наборы
    df_train, df_test = split_data(df, **preproc)

    # Выделение данных для оценки модели
    df_evaluate = df_test.copy()

    # Применение feature engineering
    df_train, df_test = generate_features(df_train, df_test, **preproc)

    # Группировка тестовых данных для рекомендации
    df_recommend_submission = df_test.groupby(rec_sub['group'])[rec_sub['values']].apply(list).to_frame(rec_sub['name'])

    # Сохранение предобработанных данных
    load_data(df_train, df_test, df_recommend_submission, df_evaluate)
