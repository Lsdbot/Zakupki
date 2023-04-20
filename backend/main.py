import spacy

from src.transforming_data.upload_data import load_data

if __name__ == '__main__':
    nlp = spacy.load("ru_core_news_sm")

    config_path = "../config/params.yaml"
    load_data(config_path)
