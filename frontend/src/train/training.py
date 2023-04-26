"""
Программа: Тренировка модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import os
import joblib
import requests
import streamlit as st
from optuna.visualization import plot_param_importances, plot_optimization_history

def training_recommender(endpoint: str):
    """
    Тренировка рекомендательной модели
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """
    with st.spinner("Модель обучается..."):
        output = requests.post(endpoint, timeout=8000)

    st.success(output)
    st.write(output.text)


def training_win_predictor(config, endpoint: str):
    """
    Тренировка модели предсказания
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """

    with st.spinner("Модель обучается..."):
        output = requests.post(endpoint, timeout=8000)

    st.success(output)
    st.write(output.text)

    # plot study
    # study = joblib.load(os.path.join(config["train"]["study_path"]))
    # fig_imp = plot_param_importances(study)
    # fig_history = plot_optimization_history(study)
    #
    # st.plotly_chart(fig_imp, use_container_width=True)
    # st.plotly_chart(fig_history, use_container_width=True)