import spacy

import os

from src.transforming_data.upload_data import load_data
from src import recommender
from src import win_predictor
from src import evaluation

from fastapi import FastAPI
import uvicorn


app = FastAPI()

@app.get("/")
async def home():
    return {"message": "Это стартовая страница пользователя"}

@app.get("/load-recommender-model/{user_id}")
async def load_recommender_model(user_id: int, config_file_location: str):
    global users, recommender_loaded_users
    if user_id not in users:
        return {"message": f"Пользователь с id={user_id} отсутствует в базе данных"}
    # Загрузка модели рекомендательной системы
    recommender.upload_data.load_model(user_id, config_file_location)
    recommender_loaded_users.append(user_id)
    return {"message": f"Модель рекомендательной системы загружена для пользователя {user_id}"}

@app.get("/load-win-predictor-model")
async def load_win_predictor_model(config_file_location: str):
    global win_predictor_loaded
    # Загрузка модели прогнозирования побед
    win_predictor.upload_data.load_model(config_file_location)
    win_predictor_loaded = True
    return {"message": "Модель прогнозирования загружена"}

@app.get("/load-evaluation-data/{user_id}")
async def load_evaluation_data(user_id: int, config_file_location: str):
    global recommender_loaded_users, win_predictor_loaded
    # Проверка, вызывалась ли функция load_recommender_model с данным id пользователя в текущем сеансе
    if user_id not in recommender_loaded_users:
        return {"message": f"Модель рекомендательной системы не загружена для пользователя {user_id}"}
    # Проверка, вызывалась ли функция load_win_predictor_model в текущем сеансе
    if not win_predictor_loaded:
        return {"message": "Модель прогнозирования побед не загружена"}
    # Загрузка данных для оценки рекомендательной системы и модели прогнозирования побед
    evaluation.upload_data.load_data(user_id, config_file_location)
    return {"message": "Данные для оценки загружены"}

@app.get("/load-all-data/{user_id}")
async def load_all_data(user_id: int, recommender_config_location: str, evaluation_config_location: str):
    # Загрузка модели рекомендательной системы и данных для оценки
    response_recommender = await load_recommender_model(user_id, recommender_config_location)
    response_evaluation = await load_evaluation_data(user_id, evaluation_config_location)
    return {"message": f"{response_recommender['message']}. {response_evaluation['message']}"}

@app.on_event("startup")
async def startup_event():
    # Запуск приложения
    print("Приложение запущено")


if __name__ == '__main__':

    users = [1, 2, 3]  # пример массива пользователей
    recommender_loaded_users = []
    win_predictor_loaded = False

    processed_data = "../data/processed"
    config_path = "../config/params.yaml"

    if not os.path.isdir(processed_data):
        nlp = spacy.load("ru_core_news_sm")

        load_data(config_path)

        del nlp

    uvicorn.run("main:app", host="127.0.0.1", port=80, reload=True)
