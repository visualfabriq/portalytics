import string
import random
import joblib
import json
import pandas as pd

from vf_portalytics.terrain_model import Terrain
import vf_portalytics.terrain_config as config
from vf_portalytics.model import PredictionModel

TEST_RANGE = 4
TEST_ACCOUNTS = ['TEST_ACCOUNT_{}'.format(i) for i in range(TEST_RANGE)]


def id_generator(size=24, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


product_index = [id_generator(6) for i in range(TEST_RANGE)]
account_index = [id_generator(7) for i in range(TEST_RANGE)]
prod_hash = [id_generator(5) for i in range(TEST_RANGE)]
coef_index = pd.MultiIndex.from_product([TEST_ACCOUNTS, prod_hash], names=[config.customer_col, config.prod_line_col])

coefs = pd.DataFrame({'discount_coef': [random.uniform(-1, -5) for i in range(len(coef_index))],
                      'display_coef': [random.uniform(0.05, 0.30) for i in range(len(coef_index))],
                      'feature_coef': [random.uniform(0.05, 0.30) for i in range(len(coef_index))],
                      }, index=coef_index
                     )

account_id_map = pd.DataFrame({config.customer_col: TEST_ACCOUNTS,
                               }, index=account_index
                              )

pid_map = pd.DataFrame({config.prod_line_col: prod_hash,
                        }, index=product_index
                       )


def generate_predict_df(n=TEST_RANGE):
    df = pd.DataFrame({
        config.account_id_col: account_index,
        config.item_col: product_index,
        config.baseline_col: [random.randint(10, 10000) for i in range(n)],
        config.discount_col: [random.uniform(0.01, 0.5) for i in range(n)],
        config.display_col: [0.0] * n,
        config.feature_col: [0.0] * n
    })

    return df


def generate_model_paths(model_data_path):
    coefs.to_parquet('{}/coefs.parquet'.format(model_data_path))
    account_id_map.to_parquet('{}/account_id_map.parquet'.format(model_data_path))
    pid_map.to_parquet('{}/pid_map.parquet'.format(model_data_path))

    coefs_path = '{}/coefs.parquet'.format(model_data_path)
    account_id_mapper_path = '{}/account_id_map.parquet'.format(model_data_path)
    pid_mapper_path = '{}/pid_map.parquet'.format(model_data_path)

    return coefs_path, account_id_mapper_path, pid_mapper_path


def test_terrain_with_prediction_model(tmpdir):
    model_data_path = str(tmpdir.mkdir('model_data'))
    coefs_path, account_id_mapper_path, pid_mapper_path = generate_model_paths(model_data_path)

    terrain_model = Terrain(coefs_path=coefs_path, file_format='parquet', account_id_mapper_path=account_id_mapper_path,
                            pid_mapper_path=pid_mapper_path)
    model_id = 'terrain_test'
    joblib.dump(terrain_model, '{0}/{1}.pkl'.format(model_data_path, model_id), compress=3)
    with open('{0}/{1}.meta'.format(model_data_path, model_id), "w") as metadata_file:
        json.dump(config.metadata, metadata_file)

    model = PredictionModel(model_id, model_data_path)

    df = generate_predict_df()
    pred = model.predict(df)

    assert not (pred == df[config.baseline_col]).all()


def test_terrain_model_predict(tmpdir):
    model_data_path = str(tmpdir.mkdir('model_data'))
    coefs_path, account_id_mapper_path, pid_mapper_path = generate_model_paths(model_data_path)

    terrain_model = Terrain(coefs_path=coefs_path, file_format='parquet', account_id_mapper_path=account_id_mapper_path,
                            pid_mapper_path=pid_mapper_path)
    df = generate_predict_df()
    pred = terrain_model.predict(df)

    assert not (pred == df[config.baseline_col]).all()


def test_get_factor(tmpdir):
    model_data_path = str(tmpdir.mkdir('model_data'))
    coefs_path, account_id_mapper_path, pid_mapper_path = generate_model_paths(model_data_path)

    terrain_model = Terrain(coefs_path=coefs_path, file_format='parquet', account_id_mapper_path=account_id_mapper_path,
                            pid_mapper_path=pid_mapper_path)

    df = generate_predict_df()
    factor = terrain_model._get_factor(df)

    assert factor != 0
