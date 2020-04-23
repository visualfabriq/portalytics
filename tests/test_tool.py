import pandas as pd
import random

from vf_portalytics.tool import get_categorical_features


def test_get_categorical_featuers():
    data = {
        'Name': [random.choice(['Tom', 'nick', 'krish', 'jack']) for i in range(100)],
        'Age': [random.choice([20, 21, 19, 18]) for i in range(100)],
        'Height': [random.uniform(1.7, 2) for i in range(100)]
    }
    data = pd.DataFrame(data)

    potential_cat_feat = get_categorical_features(data)
    cat_feat = ['Name', 'Age']
    assert set(cat_feat) == set(potential_cat_feat)

    potential_cat_feat = get_categorical_features(data, potential_cat_feat=['Height'])
    cat_feat = ['Height']
    assert set(cat_feat) == set(potential_cat_feat)