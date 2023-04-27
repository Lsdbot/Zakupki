import spacy

import os

from src.transforming_data.upload_data import load_data
from src import recommender, win_predictor, evaluation

from fastapi import FastAPI
import uvicorn


app = FastAPI()
CONFIG_PATH = "/home/sergey/projects/zakupki/config/params.yaml"


@app.get("/")
async def home():
    return "Это стартовая страница пользователя"


@app.post("/load-recommender-model/{user_id}")
async def load_recommender_model(user_id: int):
    global users, recommender_loaded_users, CONFIG_PATH

    if user_id not in users:
        return f"Пользователь с id={user_id} отсутствует в базе данных"

    # Загрузка модели рекомендательной системы
    recommender.load_model(CONFIG_PATH, user_id)
    recommender_loaded_users.append(user_id)

    return f"Модель рекомендательной системы загружена для пользователя {user_id}"


@app.post("/load-win-predictor-model")
async def load_win_predictor_model():
    global CONFIG_PATH

    # Загрузка модели прогнозирования побед
    win_predictor.load_model(CONFIG_PATH)

    return f"Модель прогнозирования загружена."


@app.post("/load-evaluation-data/recommends/{user_id}")
async def load_evaluation_recommends(user_id: int):
    global recommender_loaded_users, CONFIG_PATH

    # Проверка, вызывалась ли функция load_recommender_model с данным id пользователя в текущем сеансе
    if user_id not in recommender_loaded_users:
        return f"Модель рекомендательной системы не загружена для пользователя {user_id}"

    # Загрузка данных для оценки рекомендательной системы
    recommends = evaluation.load_recommends(CONFIG_PATH, user_id)

    return f"Данные проверки успешно загружены. \
    Рекомендуемые id закупок для пользователя {user_id} следующие: {recommends}"


@app.post("/load-evaluation-data/win_predict")
async def load_evaluation_win_predict():
    global CONFIG_PATH

    # Загрузка данных для оценки рекомендательной системы и модели прогнозирования побед
    predicts = evaluation.load_predicts(CONFIG_PATH)

    return f"Данные для оценки загружены. Результаты предсказаний следующие: {predicts[:5]}"


@app.post("/load-all-data/{user_id}")
async def load_all_data(user_id: int):
    # Загрузка модели рекомендательной системы и данных для оценки
    response_recommender = await load_recommender_model(user_id)
    response_evaluation = await load_evaluation_recommends(user_id)

    return f"{response_recommender}. {response_evaluation}"


if __name__ == '__main__':

    # Список пользователей базы данных
    users = evaluation.load_users(CONFIG_PATH)

    # Список пользователей с обученной моделью
    recommender_loaded_users = []

    # Директория предобработанных данных
    processed_data = "../data/processed"

    # Проверка на отсутствие предобработанных данных
    if not os.path.isdir(processed_data):
        nlp = spacy.load("ru_core_news_sm")

        load_data(CONFIG_PATH)

        del nlp

    # Запуск API
    uvicorn.run(app, host="127.0.0.1", port=8000)
