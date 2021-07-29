import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_regression

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


def make_regression_dataset(n_samples, n_features, n_informative, **kwargs):
    x, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=0.5,
        n_informative=n_informative,
        random_state=0
    )
    x = pd.DataFrame(x)

    x.columns = ['feature_' + str(i) for i in range(n_features)]
    x = x.assign(**kwargs)
    return x, pd.Series(y, name='target')


def make_dataset():
    # Generate data for 4 different categories
    # different #samples for each category but the same #features since they belong to the same dataset
    n_features = 5
    x1, y1 = make_regression_dataset(n_samples=100, n_features=n_features, n_informative=3, category='A')
    x2, y2 = make_regression_dataset(n_samples=150, n_features=n_features, n_informative=4, category='B')
    x3, y3 = make_regression_dataset(n_samples=80, n_features=n_features, n_informative=5, category='C')
    x4, y4 = make_regression_dataset(n_samples=120, n_features=n_features, n_informative=1, category='D')

    # combine into one dataset
    total_x = pd.concat([x1, x2, x3, x4], axis=0, ignore_index=True).reset_index(drop=True)
    total_y = pd.concat([y1, y2, y3, y4], axis=0, ignore_index=True).reset_index(drop=True)

    # make two random features categorical
    labels = ['g1', 'g2', 'g3']
    bins = [[], []]
    for i in range(2):
        bins[i] = [-np.inf,
                   total_x['feature_' + str(i)].mean() - total_x['feature_' + str(i)].std(),
                   total_x['feature_' + str(i)].mean() + total_x['feature_' + str(i)].std(),
                   total_x['feature_' + str(i)].max()]
    total_x['feature_0'] = pd.cut(total_x['feature_0'], bins=bins[0], labels=labels).astype('object')
    total_x['feature_1'] = pd.cut(total_x['feature_1'], bins=bins[1], labels=labels).astype('object')

    return total_x, total_y
