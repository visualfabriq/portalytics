import string
import random
import joblib
import json
import pandas as pd

from vf_portalytics.terrain_model import Terrain
from vf_portalytics.terrain_config import metadata
from vf_portalytics.model import PredictionModel


def id_generator(size=24, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def generate_predict_df(n):
    df = pd.DataFrame({
        'promo': [random.randrange(0, n / 2, 1) for i in range(n)],
        'account_id': ['SIMPLIFIED'] * n,
        'pid': [id_generator(6) for i in range(n)],
        'consumer_length': [random.randint(1, 5) for i in range(n)],
        'baseline_units_ext': [random.randint(10, 10000) for i in range(n)],
        'discount_perc': [random.randrange(10, 50, 10) for i in range(n)],
        'second_placement_perc': [random.choice([0, 50, 60, 70]) for i in range(n)],
        'feature_perc': [random.choice([0, 50]) for i in range(n)],
    })

    return df


def generate_model_paths(model_data_path):
    coefs.to_csv(f'{model_data_path}/coefs.csv')
    account_id_map.to_csv(f'{model_data_path}/account_id_map.csv')
    pid_map.to_csv(f'{model_data_path}/pid_map.csv')

    coefs_path = f"{model_data_path}/coefs.csv"
    account_id_mapper_path = f"{model_data_path}/account_id_map.csv"
    pid_mapper_path = f"{model_data_path}/pid_map.csv"

    return coefs_path, account_id_mapper_path, pid_mapper_path


coefs = pd.DataFrame({'discount_coef': [-1.1, -2.2, -3.3, -1.8, -2.1],
                      'display_coef': [0.0, 0.0, 0.1, 0.13, 0.12],
                      'feature_coef': [0.0, 0.0, 0.01, 0.11, 0.012],
                      }, index=['SIMPLIFIED'] * 5
                     )

account_id_map = pd.DataFrame({'cpe_1': ['SIMPLIFIED'] * 5,
                               }, index=[id_generator() for i in range(5)]
                              )

pid_map = pd.DataFrame({'prod_line_hash': [id_generator(10) for i in range(5)],
                        }, index=[id_generator() for i in range(5)]
                       )


def test_terrain_model_predict(tmpdir):
    model_data_path = str(tmpdir.mkdir('model_data'))
    coefs_path, account_id_mapper_path, pid_mapper_path = generate_model_paths(model_data_path)

    terrain_model = Terrain(coefs_path=coefs_path, file_format='csv', account_id_mapper_path=account_id_mapper_path,
                            pid_mapper_path=pid_mapper_path)
    model_id = 'terrain_test'
    joblib.dump(terrain_model, f"{model_data_path}/{model_id}.pkl", compress=3)
    with open(f"{model_data_path}/{model_id}.meta", "w") as metadata_file:
        json.dump(metadata, metadata_file)

    model = PredictionModel(model_id, model_data_path)

    df = generate_predict_df(50)
    pred = model.predict(df)

    assert not (pred == 0).all()
