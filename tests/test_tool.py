import pandas as pd
import random

from vf_portalytics.tool import set_categorical_features


def test_set_categorical_featuers():
    data = {
        'Name': [random.choice(['Tom', 'nick', 'krish', 'jack']) for i in range(100)],
        'Age': [random.choice([20, 21, 19, 18]) for i in range(100)],
        'Height': [random.uniform(1.7, 2) for i in range(100)]
    }
    data = pd.DataFrame(data)

    potential_cat_feat = set_categorical_features(data)
    cat_feat = set(['Name', 'Age'])
    assert cat_feat == potential_cat_feat

    potential_cat_feat = set_categorical_features(data, potential_cat_feat=['Height'])
    cat_feat = set(['Height'])
    assert cat_feat == potential_cat_feat