"""
Программа: Frontend часть проекта
Версия: 1.0
"""


from src import charts, evaluation, data, train
import pandas as pd
import streamlit as st
import yaml

CONFIG_PATH = "../config/params.yaml"


def home():
    """
    Функция для создания стартовой страницы с описанием проекта.
    """

    # заголовок страницы
    st.markdown("# Описание проекта")
    st.title("MLOps project: Recommender system for suppliers in trading platform")

    # описание проекта
    st.write("""Оператор электронных торгов предоставляет торговую площадку для государственных заказчиков,
             госкомпаний и коммерческих предприятий. Им необходима рекомендательная система, которая бы обеспечила 
             поставщикам удобную навигацию по сервису, преоритетно показывая релевантные для них торги.""")

    # список признаков датасета
    st.markdown(
        """
        ### Описание полей 
        
        Поля в датасете purchases: 
        
            - purchase – Уникальный идентификатор закупки
            - region_code – Регион поставки
            - min_publish_date – Дата публикации закупки 
            - purchase_name – Название закупки
            - forsmallbiz – Если да, то большие компании не могут предлагать свое участие в этой закупке
            - price – Цена за закупку, предлагаемая заказчиком
            - customer – Уникальный идентификатор заказчика
            - okpd2_code – Код стандартизации товара в соответствии со справочником ОКПД2, обрезанный до 3 числа
            - okpd2_names – Названия ОКПД2
            - Item_descriptions – Описания товаров
            
        Поля в датасете suppliers: 
        
            - purchase – Уникальный идентификатор закупки
            - supplier – Уникальный идентификатор поставщика
            - is_winner – Выиграл ли этот поставщик в этой закупке

        """
    )


def eda():
    """
    Функция для создания страницы разведочного анализа данных с графиками.
    """
    # заголовок страницы
    st.title("Разведочный анализ данных")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # параметры конфигурации предобработки
    preproc = config['preprocessing']

    # загрузка данных
    df = data.get_data(preproc['train_data'], 'vectorized')
    st.write(df[:5])


    uniq_okpd2 = st.sidebar.checkbox("Кол-во уникальных ОКПД2 у поставщиков")
    season_activity = st.sidebar.checkbox("Активность поставщиков в течение года")
    price_mean_std = st.sidebar.checkbox("Средняя цена и разброс цены закупок поставщиков")


    if uniq_okpd2:
        st.pyplot(
            charts.chart_unique_okpd2(df, 'n_unique_okpd2',
                                      title="Плотность распределения кол-ва уникальных ОКПД2 у поставщиков")
        )

    if season_activity:
        ax = charts.chart_season_activity(df, 'month', 'supplier',
                                          title="Кол-во участий поставщиков в разрезе месяца")

        fig = ax.get_figure()

        st.pyplot(fig)

    if price_mean_std:
        ax = charts.chart_price(df, 'supplier', 'price',
                                title='Средняя цена - Разброс цены закупки у поставщиков')

        fig = ax.get_figure()

        st.pyplot(fig)


def load_evaluate_data():
    """
    Функция для загрузки csv файла и преобразования его в датафрейм.
    """
    # заголовок страницы
    st.title("Загрузка файла")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # форма для загрузки файла
    uploaded_file = st.file_uploader("Выберите csv файл", type="csv")

    # проверка наличия загруженного файла
    if uploaded_file is not None:
        # преобразование файла в датафрейм
        df = pd.read_csv(uploaded_file)

        # вывод датафрейма на страницу
        st.write(df[:5])


def train_recommender():
    """
    Функция для создания страницы тренировки рекомендательной модели.
    """
    # заголовок страницы
    st.title("Тренировка рекомендательной модели")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # ввод id поставщика
    supplier_id = st.text_input("Введите id поставщика")

    if supplier_id:
        endpoint = config['endpoints']['train_recommender'] + str(supplier_id)

        # кнопка для запуска тренировки модели
        if st.button("Запустить тренировку"):
            # тренировка модели
            train.training_recommender(endpoint)


def train_win_predictor():
    """
    Функция для создания страницы тренировки прогнозирующей модели.
    """
    # заголовок страницы
    st.title("Тренировка прогнозирующей модели")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    endpoint = config['endpoints']['train_win_predictor']

    # кнопка для запуска тренировки модели
    if st.button("Запустить тренировку"):
        # тренировка модели
        train.training_win_predictor(config, endpoint)


def get_recommends():
    """
    Получение предсказаний рекомендательной модели
    """
    # заголовок страницы
    st.title("Получение предсказаний рекомендательной модели")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # ввод id поставщика
    supplier_id = st.text_input("Введите id поставщика")

    if supplier_id:
        endpoint = config["endpoints"]["evaluate_recommender"] + str(supplier_id)

        if st.button("Запустить подбор рекомендаций"):
            evaluation.evaluate_recommender(endpoint)


def get_predicts():
    """
    Получение предсказаний прогнозирующей модели
    """
    # заголовок страницы
    st.title("Получение предсказаний прогнозирующей модели")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    endpoint = config["endpoints"]["evaluate_win_predictor"]

    if st.button("Запустить подбор рекомендаций"):
        evaluation.evaluate_win_predictor(endpoint)


def get_metrics():
    """
    Вывод метрик качества
    """
    # заголовок страницы
    st.title("Метрики качества")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config['train']['recommender']['metrics']) as file:
        recommender_metrics = yaml.load(file, Loader=yaml.FullLoader)

    with open(config['train']['win_predictor']['metrics']) as file:
        win_predictor_metrics = yaml.load(file, Loader=yaml.FullLoader)


    dict_metrics = {
        "Базовые метрики модели рекомендаций": recommender_metrics['basic_metrics'],
        "Лучшие метрики модели рекомендаций": recommender_metrics['best_metrics'],
        "Базовые метрики модели прогнозирования": win_predictor_metrics['basic_metrics']['catboost'],
        "Лучшие метрики модели прогнозирования": win_predictor_metrics['best_metrics']['catboost']
    }

    type_metric = st.sidebar.selectbox("Метрики качества", dict_metrics.keys())
    st.write(dict_metrics[type_metric])


def main():
    """
    Сборка пайплайна в одном блоке
    """
    pages_to_funcs = {
        "Описание проекта": home,
        "Разведочный анализ данных": eda,
        "Тренировка рекомендательной модели": train_recommender,
        "Тренировка прогнозирующей модели": train_win_predictor,
        "Получение рекомендаций": get_recommends,
        "Получение прогнозов победителя": get_predicts,
        "Метрики качества моделей": get_metrics
    }
    page = st.sidebar.selectbox("Выберите пункт", pages_to_funcs.keys())
    pages_to_funcs[page]()


if __name__ == "__main__":
    main()