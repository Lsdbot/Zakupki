"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def remove_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Функция удаляет выбросы из DataFrame по заданному столбцу.

    Аргументы:
        df (pd.DataFrame): Исходный DataFrame
        col (str): Название столбца, по которому будут удалены выбросы

    Возвращает:
        pd.DataFrame: DataFrame без выбросов
    """
    mean = df[col].mean()
    std = df[col].std()

    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    return df[(df[col] > lower_bound) & (df[col] < upper_bound)]


def chart_unique_okpd2(df: pd.DataFrame, column: str, title: str) -> matplotlib.figure.Figure:
    """
    Функция строит KDE-график распределения уникальных значений столбца column из DataFrame df.

    Аргументы:
        df (pd.DataFrame): Исходный DataFrame
        column (str): Название столбца, для которого будет построен график
        title (str): Заголовок графика

    Возвращает:
        matplotlib.figure.Figure: Объект графика
    """
    fig = sns.displot(x=df[column], kind='kde')

    plt.title(title, fontsize=14)

    return fig


def chart_season_activity(df: pd.DataFrame, x_column: str, y_column: str, title: str) -> matplotlib.figure.Figure:
    """
    Функция строит столбчатую диаграмму для количества значений y_column,
    сгруппированных по значениям x_column в DataFrame df.

    Аргументы:
        df (pd.DataFrame): Исходный DataFrame
        x_column (str): Название столбца, по которому будут сгруппированы значения
        y_column (str): Название столбца, для которого будут построены столбцы диаграммы
        title (str): Заголовок диаграммы

    Возвращает:
        matplotlib.figure.Figure: Объект диаграммы
    """
    plt.figure(figsize=(12, 7))

    df_temp = df.groupby(x_column)[y_column].size()

    ax = sns.barplot(x=df_temp.index, y=df_temp, palette='rocket')

    ax.set_title(title, fontsize=14)

    return ax


def chart_price(df: pd.DataFrame, x_column: str, y_column: str, title: str) -> matplotlib.figure.Figure:
    """
    Функция строит диаграмму рассеяния, используя столбцы x_column и y_column DataFrame df.
    Вычисляется среднее значение и стандартное отклонение для столбца y_column для каждого уникального
    значения x_column. Затем значения, выходящие за пределы 3-х стандартных отклонений, удаляются.

    Аргументы:
        df (pd.DataFrame): Исходный DataFrame
        x_column (str): Название столбца, отображаемого на оси x
        y_column (str): Название столбца, отображаемого на оси y
        title (str): Заголовок диаграммы

    Возвращает:
        matplotlib.figure.Figure: Объект диаграммы
    """
    # датафрейм со средней ценой и разбросом цены закупок у поставщика
    df_temp = df.groupby(x_column)[y_column].mean().to_frame(name='mean')
    df_temp['std'] = df.groupby(x_column)[y_column].std()

    # удаление выбросов
    df_temp = remove_outliers(df_temp, 'mean')
    df_temp = remove_outliers(df_temp, 'std')

    plt.figure(figsize=(12, 7))

    ax = sns.scatterplot(x=df_temp['mean'], y=df_temp['std'])

    ax.set_title(title, fontsize=14)

    return ax
