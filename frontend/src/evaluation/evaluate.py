"""
Программа: Отрисовка слайдеров и кнопок для запуска получения предсказаний
Версия: 1.0
"""

import streamlit as st
import requests


def evaluate_recommender(endpoint: str) -> None:
    """
    Оценка рекомендательной системы.

    Аргументы:
    endpoint -- URL-адрес эндпоинта.

    Возвращает:
    None.
    """
    # заголовки запроса
    headers = {
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36 OPR/40.0.2308.81',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'DNT': '1',
        'Accept-Encoding': 'gzip, deflate, lzma, sdch',
        'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.6,en;q=0.4'
    }

    with st.spinner('Модель высчитывает результаты...'):
        output = requests.get(endpoint, timeout=20000, headers=headers)

    st.success(output)
    st.write(output.text)


def evaluate_win_predictor(endpoint: str):
    """
    Оценка модели прогнозирования побед.

    Аргументы:
    endpoint -- URL-адрес эндпоинта.

    Возвращает:
    None.
    """
    # заголовки запроса
    headers = {
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36 OPR/40.0.2308.81',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'DNT': '1',
        'Accept-Encoding': 'gzip, deflate, lzma, sdch',
        'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.6,en;q=0.4'
    }

    with st.spinner('Модель высчитывает результаты...'):
        output = requests.get(endpoint, timeout=20000, headers=headers)

    st.success(output)
    st.write(output.text)
