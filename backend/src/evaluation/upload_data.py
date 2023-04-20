from .pipeline import pipeline_evaluate

import yaml


def load_data(config_path, supplier_id):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return pipeline_evaluate(config, supplier_id)