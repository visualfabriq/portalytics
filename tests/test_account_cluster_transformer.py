import pandas as pd
import random

from vf_portalytics.ml_helpers import AccountClusterTransformer


def test_prediction_model_account_clusters():
    train_x = {
        'account_banner': [random.choice(['A', 'B', 'C', 'D']) for i in range(100)],
        'Var_1': [random.choice([22, 21, 19, 18]) for i in range(100)]
    }
    train_y = {
        'Target_1': [random.uniform(1.7, 2) for i in range(100)]
    }
    train_x = pd.DataFrame(train_x)
    train_y = pd.DataFrame(train_y)

    cat_feature = ['A', 'B', 'C', 'D']
    transformer = AccountClusterTransformer(cat_feature)
    transformed_x = transformer.fit_transform(train_x, train_y)

    assert all(train_x['Var_1'] == transformed_x['Var_1'])

    assert ('cluster' in transformed_x.columns)
