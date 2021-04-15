import pandas as pd
import random

import pytest

from vf_portalytics.ml_helpers import CustomClusterTransformer


def make_dataset():
    train_x = {
        'account_banner': [random.choice(['A', 'B', 'C', 'D']) for i in range(100)],
        'Var_1': [random.choice([22, 21, 19, 18]) for i in range(100)]
    }
    train_y = {
        'Target_1': [random.uniform(1.7, 2) for i in range(100)]
    }
    train_x = pd.DataFrame(train_x)
    train_y = pd.DataFrame(train_y)

    return train_x, train_y


def test_prediction_model_custom_clusters():
    # Case 1
    train_x_c1, train_y_c1 = make_dataset()
    cat_feature = 'account_banner'
    transformer_c1 = CustomClusterTransformer(cat_feature)
    transformed_x_c1 = transformer_c1.fit_transform(train_x_c1, train_y_c1)

    assert all(train_x_c1['Var_1'] == transformed_x_c1['Var_1'])

    assert all(transformed_x_c1['cluster'] != 0.0)

    # Case 2
    train_x_c2, train_y_c2 = make_dataset()
    cat_feature_unknown = 'Unknown_Var'
    transformer_c2 = CustomClusterTransformer(cat_feature_unknown)
    with pytest.raises(KeyError):
        transformer_c2.fit_transform(train_x_c2, train_y_c2)

    # Case 3
    train_x_c3, train_y_c3 = make_dataset()
    cat_feature_vf = 'vf_category'
    transformer_c3 = CustomClusterTransformer(cat_feature_vf)
    transformed_x_c3 = transformer_c3.fit_transform(train_x_c3, train_y_c3)

    assert all(transformed_x_c3['cluster'] == 0.0)
