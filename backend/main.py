import ru_core_news_sm

import os

from src.transforming_data.upload_data import load_data
from src import recommender, win_predictor, evaluation

from fastapi import FastAPI
import uvicorn


app = FastAPI()
CONFIG_PATH = "../config/params.yaml"

USERS = evaluation.load_users(CONFIG_PATH)

RECOMMENDER_LOADED_USERS = []


@app.get("/")
async def home():
    return "Это стартовая страница пользователя"


@app.get("/load-recommender-model/{user_id}")
async def load_recommender_model(user_id: int):
    global USERS, RECOMMENDER_LOADED_USERS, CONFIG_PATH

    if user_id not in USERS:
        return f"Пользователь с id={user_id} отсутствует в базе данных"

    # Загрузка модели рекомендательной системы
    recommender.load_model(CONFIG_PATH, user_id)
    RECOMMENDER_LOADED_USERS.append(user_id)

    return f"Модель рекомендательной системы загружена для пользователя {user_id}"


@app.get("/load-win-predictor-model")
async def load_win_predictor_model():
    global CONFIG_PATH

    # Загрузка модели прогнозирования побед
    win_predictor.load_model(CONFIG_PATH)

    return f"Модель прогнозирования загружена."


@app.get("/load-evaluate-recommends/{user_id}")
async def load_evaluation_recommends(user_id: int):
    global RECOMMENDER_LOADED_USERS, CONFIG_PATH

    # Проверка, вызывалась ли функция load_recommender_model с данным id пользователя в текущем сеансе
    if user_id not in RECOMMENDER_LOADED_USERS:
        return f"Модель рекомендательной системы не загружена для пользователя {user_id}"

    # Загрузка данных для оценки рекомендательной системы
    recommends = evaluation.load_recommends(CONFIG_PATH, user_id)

    if len(recommends) < 6:
        return f"Рекомендуемые id закупок для пользователя {user_id} следующие: {recommends}"

    return f"Рекомендуемые id закупок для пользователя {user_id} следующие: {recommends[:5]}"


@app.get("/load-evaluate-win-predict")
async def load_evaluation_win_predict():
    global CONFIG_PATH

    # Загрузка данных для оценки рекомендательной системы и модели прогнозирования побед
    predicts = evaluation.load_predicts(CONFIG_PATH)

    return f"Данные для оценки загружены. Результаты предсказаний следующие: {predicts[:5]}"


@app.get("/load-all-data/{user_id}")
async def load_all_data(user_id: int):
    # Загрузка модели рекомендательной системы и данных для оценки
    response_recommender = await load_recommender_model(user_id)
    response_evaluation = await load_evaluation_recommends(user_id)

    return f"{response_recommender}. {response_evaluation}"


if __name__ == '__main__':

    # Директория предобработанных данных
    processed_data = "../data/processed"

    # Проверка на отсутствие предобработанных данных
    if not os.path.isdir(processed_data):
        nlp = ru_core_news_sm.load()

        load_data(CONFIG_PATH)

        del nlp

    # Запуск API
    uvicorn.run(app, host="127.0.0.1", port=8000)
