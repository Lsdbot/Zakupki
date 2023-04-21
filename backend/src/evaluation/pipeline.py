from .get_data import get_data, get_recommend_model
from .evaluate import get_supplier_data, generate_features, transform_vector


def pipeline_evaluate(config, supplier_id):
    preproc = config['preprocessing']
    train_recommender = config['train']['recommender']
    evaluate = config['evaluate']

    train_data = get_data(preproc['train_data'], train_recommender['vector'])
    evaluate_data = get_data(evaluate['evaluate_data'], train_recommender['vector'])

    evaluate_data = generate_features(train_data, evaluate_data, preproc)

    test_purchases = evaluate_data[evaluate_data[preproc['recommender']['sup_column']] == supplier_id][preproc['recommender']['index_column']]

    evaluate_data = transform_vector(evaluate_data, n_components=preproc['n_components'],
                                     vector=train_recommender['vector'])

    evaluate_sup_data = get_supplier_data(train_data, evaluate_data, supplier_id,
                                          test_purchases, **train_recommender)

    model_recommender = get_recommend_model(train_recommender['models'], supplier_id)

    y_pred = model_recommender.predict(evaluate_sup_data)

    return evaluate_sup_data[y_pred == 1].index.tolist()
