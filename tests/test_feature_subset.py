import itertools
import random

import numpy as np
import numpy.testing as npt
import pandas as pd
from numpy.random import randint
from pandas.util.testing import assert_series_equal
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from vf_portalytics.feature_subset import FeatureSubsetModel, FeatureSubsetTransform
from vf_portalytics.model import PredictionModel


def make_dataset(random_state, n_informative, collumn_names, **kwargs):
    x, y = make_regression(

        n_samples=1000,
        n_features=5,
        noise=0 if random_state == 1 else 10,
        bias=10 if random_state == 1 else 1000,
        n_informative=min(n_informative, 5),
        random_state=random_state
    )
    x = pd.DataFrame(x)
    x.columns = [name for name in collumn_names]
    x = x.assign(**kwargs)
    x['yearweek'] = randint(1, 54, len(x))
    # pack_type (original_product_dimension_44) 0: 'Can', 1: 'Bottle'
    x['original_product_dimension_44'] = [random.choice([0, 1]) for i in range(len(x))]
    return x, pd.Series(y)


def make_dict():
    """Creates a dictionary with keys all the combinations between the weeks of the year and the pack types
    In this case we have only one pack type, in order to check if when we dont have the pack type in the dict,
    the model will predict 0.
    """
    all_list = [list(range(1, 54)), [0]]
    keys = list(itertools.product(*all_list))
    values = [random.choice(np.linspace(-2.5, 2.5, num=500)) for i in range(len(keys))]
    return dict(zip(keys, values))

def test_feauture_subset_model():
    collumn_names = ['promoted_price', 'consumer_length',
                     'yearweek', 'original_product_dimension_44', 'product_volume_per_sku']

    x1, y1 = make_dataset(1, 5, collumn_names, account_banner='A', product_desc='X')
    x2, y2 = make_dataset(2, 3, collumn_names, account_banner='B', product_desc='Y')
    # create on more that will not have sub_model and will predict 0
    x3, y3 = make_dataset(3, 1, collumn_names, account_banner='C', product_desc='Z')

    # combine into one dataset
    total_x = pd.concat([x1, x2, x3], axis=0, ignore_index=True).reset_index(drop=True)
    total_y = pd.concat([y1, y2, y3], axis=0, ignore_index=True).reset_index(drop=True)
    # Split into train and test
    train_index, test_index = train_test_split(total_x.index, random_state=5)
    train_x, train_y = total_x.loc[train_index, :], total_y.loc[train_index]
    test_x, test_y = total_x.loc[test_index, :], total_y.loc[test_index]

    # create dictionary "predicted_market_volumes" - "lookup_dict"
    lookup_dict = make_dict()

    subset_cols = ('account_banner', 'product_desc')
    sub_models = {
        ('A', 'X'): LinearRegression(),
        ('B', 'Y'): DecisionTreeRegressor(),
    }

    pipeline = Pipeline([
        ('transform', FeatureSubsetTransform(group_cols=subset_cols, transformer=PolynomialFeatures(2))),
        ('estimate', FeatureSubsetModel(lookup_dict=lookup_dict, group_cols=subset_cols, sub_models=sub_models))
    ])

    # Note: must use one_hot_encode=False to prevent one-hot encoding of categorical features in input data
    model_wrapper = PredictionModel("my_test_model", path='/tmp', one_hot_encode=False)

    model_wrapper.model = pipeline
    # save feature names (no strictly since all the preprocessing is made being made in the pipeline)
    model_wrapper.features = {
        # Grouping features
        'account_banner': [],
        'product_desc': [],
        # other feaures
        'promoted_price': [],
        'consumer_length': [],
        'yearweek': [],
        'original_product_dimension_44': [],
        'product_volume_per_sku': [],
    }
    model_wrapper.target = {'target': []}
    model_wrapper.ordered_column_list = sorted(model_wrapper.features.keys())
    model_wrapper.model.fit(train_x, train_y)

    predicted_y = model_wrapper.model.predict(test_x)

    model_wrapper.model = pipeline
    _ = model_wrapper.model.fit(train_x, train_y)
    model_wrapper.save()

    # Load model and check if the properties are saved as well
    saved_model = PredictionModel('my_test_model', path='/tmp')
    saved_model.model
    saved_model.features

    saved_model_predicted_y = saved_model.model.predict(test_x)

    assert saved_model.features == model_wrapper.features
    assert saved_model.ordered_column_list == model_wrapper.ordered_column_list
    assert saved_model.target == model_wrapper.target
    assert_series_equal(saved_model_predicted_y, predicted_y, check_less_precise=0)

    # check totally new data
    # create on more that will not have sub_model and will predict 0
    x, y = make_dataset(3, 1, collumn_names, account_banner='EE', product_desc='QQ')
    predicted_y = saved_model.model.predict(x)

    npt.assert_almost_equal(predicted_y, [0] * len(predicted_y))
