"""
Программа: Отрисовка слайдеров и кнопок для запуска получения предсказаний
Версия: 1.0
"""

import streamlit as st
import requests


def evaluate_recommender(endpoint: str):
    with st.spinner('Модель высчитывает результаты...'):
        output = requests.post(endpoint)

    st.success(output)
    st.write(output.text)


def evaluate_win_predictor(endpoint: str):
    with st.spinner('Модель высчитывает результаты...'):
        output = requests.post(endpoint)

    st.success(output)
    st.write(output.text)