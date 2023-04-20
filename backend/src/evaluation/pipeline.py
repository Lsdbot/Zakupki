from .get_data import get_data, get_win_model, get_recommend_model
from .evaluate import get_supplier_data, generate_features, transform_vector


def pipeline_evaluate(config, supplier_id):
    preproc = config['preprocessing']
    train_recommender = config['train']['recommender']
    train_win_predictor = config['train']['win_predictor']
    evaluate = config['evaluate']

    train_data = get_data(preproc['train_data'], preproc['vector'])
    evaluate_data = get_data(evaluate['evaluate_data'], preproc['vector'])

    evaluate_sup_data = get_supplier_data(train_data, evaluate_data, supplier_id, **train_recommender)

    evaluate_sup_data = generate_features(train_data, evaluate_sup_data, preproc)

    evaluate_recommend = transform_vector(evaluate_sup_data, **preproc)

    model_recommender = get_recommend_model(train_recommender['models'], supplier_id)
    model_win_predictor = get_win_model(train_win_predictor['models'])

    y_pred = model_recommender.predict(evaluate_recommend.drop(
        columns=train_recommender['drop_columns']).drop_duplicates)

    recommends = evaluate_sup_data[y_pred == 1]

    probabilities = model_win_predictor.predcit_proba(recommends.drop(
        columns=train_win_predictor['drop_columns']))[:, 1]

    return recommends.index.tolist(), probabilities
