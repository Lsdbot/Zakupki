"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def remove_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    # фильтрация выбросов
    mean = df[col].mean()
    std = df[col].std()

    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    return df[(df[col] > lower_bound) & (df[col] < upper_bound)]


def chart_unique_okpd2(df: pd.DataFrame, column: str, title: str) -> matplotlib.figure.Figure:

    fig = sns.displot(x=df[column], kind='kde')

    plt.title(title, fontsize=14)

    return fig


def chart_season_activity(df: pd.DataFrame, x_column: str, y_column: str, title: str) -> matplotlib.figure.Figure:

    plt.figure(figsize=(12, 7))

    df_temp = df.groupby(x_column)[y_column].size()

    ax = sns.barplot(x=df_temp.index, y=df_temp, palette='rocket')

    ax.set_title(title, fontsize=14)

    return ax


def chart_price(df: pd.DataFrame, x_column: str, y_column: str, title: str) -> matplotlib.figure.Figure:

    df_temp = df.groupby(x_column)[y_column].mean().to_frame(name='mean')
    df_temp['std'] = df.groupby(x_column)[y_column].std()

    df_temp = remove_outliers(df_temp, 'mean')
    df_temp = remove_outliers(df_temp, 'std')

    plt.figure(figsize=(12, 7))

    ax = sns.scatterplot(x=df_temp['mean'], y=df_temp['std'])

    ax.set_title(title, fontsize=14)

    return ax
