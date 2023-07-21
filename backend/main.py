"""
Программа: Backend часть проекта
Версия: 1.0
"""

import os

from fastapi import FastAPI
import uvicorn

from src.data.get_data import get_users
from src.pipelines.pipelines import pipeline_train_recommender, pipeline_train_win_predictor
from src.transform_data.transform import pipeline_preprocessing
from src.evaluate_data.evaluate import pipeline_evaluate_predicts, pipeline_evaluate_recommends

app = FastAPI()
CONFIG_PATH = "../config/params.yaml"

USERS = get_users(CONFIG_PATH)

RECOMMENDER_LOADED_USERS = []


@app.get("/")
async def home():
    return "Это стартовая страница пользователя"


@app.get("/load-recommender-model/{user_id}")
async def load_recommender_model(user_id: int):
    """
    Загружает модель рекомендательной системы для заданного пользователя.

    Args
        user_id (int): идентификатор пользователя.

    Returns
        str: Результат загрузки модели: сообщение об успешной загрузке или об отсутствии пользователя в базе данных.
    """
    global USERS, RECOMMENDER_LOADED_USERS, CONFIG_PATH

    if user_id not in USERS:
        return f"Пользователь с id={user_id} отсутствует в базе данных"

    # обучаем рекомендательную модель для поставщика
    pipeline_train_recommender(CONFIG_PATH, user_id)

    # Загрузка модели рекомендательной системы
    RECOMMENDER_LOADED_USERS.append(user_id)

    return f"Модель рекомендательной системы загружена для пользователя {user_id}"


@app.get("/load-win-predictor-model")
async def load_win_predictor_model():
    """
    Загружает модель прогнозирования побед.

    Returns
        str: Результат загрузки модели: сообщение об успешной загрузке.
    """
    global CONFIG_PATH

    # Загрузка модели прогнозирования побед
    pipeline_train_win_predictor(CONFIG_PATH)

    return f"Модель прогнозирования загружена."


@app.get("/load-evaluate-data-recommends/{user_id}")
async def load_evaluation_recommends(user_id: int):
    """
    Загружает рекомендации для оценки качества модели рекомендательной системы для указанного пользователя.

    Args:
        user_id (int): Идентификатор пользователя, для которого загружаются рекомендации.

    Returns:
        str: Результат выполнения функции в формате строки.
             Если модель рекомендательной системы не была загружена для пользователя user_id, возвращает сообщение
             об ошибке.
             Если количество рекомендаций меньше 6, возвращает все рекомендации для пользователя user_id.
             Если количество рекомендаций больше или равно 6, возвращает первые 5 рекомендаций для пользователя user_id.
    """
    global RECOMMENDER_LOADED_USERS, CONFIG_PATH

    # Проверка, вызывалась ли функция load_recommender_model с данным id пользователя в текущем сеансе
    if user_id not in RECOMMENDER_LOADED_USERS:
        return f"Модель рекомендательной системы не загружена для пользователя {user_id}"

    # Загрузка данных для оценки рекомендательной системы
    recommends = pipeline_evaluate_recommends(CONFIG_PATH, user_id)

    if len(recommends) < 6:
        return f"Рекомендуемые id закупок для пользователя {user_id} следующие: {recommends}"

    return f"Рекомендуемые id закупок для пользователя {user_id} следующие: {recommends[:5]}"


@app.get("/load-evaluate-data-win-predict")
async def load_evaluation_win_predict():
    """
    Загружает данные для оценки качества модели прогнозирования побед.

    Returns:
        str: Результат выполнения функции в формате строки.
             Возвращает первые 5 результатов предсказаний для модели прогнозирования побед.
    """
    global CONFIG_PATH

    # Загрузка данных для оценки и модели прогнозирования побед
    predicts = pipeline_evaluate_predicts(CONFIG_PATH)

    return f"Данные для оценки загружены. Результаты предсказаний следующие: {predicts[:5]}"


@app.get("/load-all-data/{user_id}")
async def load_all_data(user_id: int):
    """
    Загружает модель рекомендательной системы и данные для оценки качества моделей.

    Args:
        user_id (int): Идентификатор пользователя, для которого загружается модель рекомендательной системы.

    Returns:
        str: Результат выполнения функции в формате строки.
             Возвращает сообщение о том, что модель рекомендательной системы загружена для пользователя user_id и
             первые 5 рекомендаций.
    """
    # Загрузка модели рекомендательной системы и данных для оценки
    response_recommender = await load_recommender_model(user_id)
    response_evaluation = await load_evaluation_recommends(user_id)

    return f"{response_recommender}. {response_evaluation}"


if __name__ == '__main__':

    # Директория предобработанных данных
    processed_data = "../data/processed"

    # Проверка на отсутствие предобработанных данных
    if not os.path.isdir(processed_data):
        # создание предобработанных данных
        pipeline_preprocessing(CONFIG_PATH)

    # Запуск API
    uvicorn.run(app, host="127.0.0.1", port=8000)
