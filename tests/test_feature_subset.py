import pandas as pd
from pandas.util.testing import assert_series_equal

import pytest

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from vf_portalytics.feature_subset import FeatureSubsetModel, FeatureSubsetTransform
from vf_portalytics.model import PredictionModel


def make_dataset(random_state, n_informative, **kwargs):
    x, y = make_regression(

        n_samples=1000,
        n_features=5,
        noise=0 if random_state == 1 else 10,
        bias=10 if random_state == 1 else 1000,
        n_informative=min(n_informative, 5),
        random_state=random_state
    )
    x = pd.DataFrame(x)
    x.columns = ['feature_{}'.format(n) for n in x.columns]
    x = x.assign(**kwargs)

    return x, pd.Series(y)

def test_feauture_subset_model():

    # Create data
    x1, y1 = make_dataset(1, 5, account_banner='A', product_desc='X')
    x2, y2 = make_dataset(2, 3, account_banner='B', product_desc='Y')

    # combine into one dataset
    total_x = pd.concat([x1, x2], axis=0, ignore_index=True).reset_index(drop=True)
    total_y = pd.concat([y1, y2], axis=0, ignore_index=True).reset_index(drop=True)

    # Split into train and test
    train_index, test_index = train_test_split(total_x.index, random_state=5)
    train_x, train_y = total_x.loc[train_index, :], total_y.loc[train_index]
    test_x, test_y = total_x.loc[test_index, :], total_y.loc[test_index]

    model_wrapper = PredictionModel("my_test_model", path='/tmp', one_hot_encode=False)
    subset_cols = ('account_banner', 'product_desc')
    sub_models = {
        ('A', 'X'): LinearRegression(),
        ('B', 'Y'): DecisionTreeRegressor(),
    }

    pipeline = Pipeline([
      ('transform', FeatureSubsetTransform(group_cols=subset_cols, transformer=PolynomialFeatures(2))),
      ('estimate', FeatureSubsetModel(group_cols=subset_cols, sub_models=sub_models))
    ])

    model_wrapper.model = pipeline
    model_wrapper.features = {
        # Grouping features
        'account_banner': [],
        'product_desc': [],
        # other feaures
        'feature_0': [],
        'feature_1': [],
        'feature_2': [],
        'feature_3': [],
        'feature_4': [],
    }

    model_wrapper.target = {'target': []}
    model_wrapper.ordered_column_list = sorted(model_wrapper.features.keys())
    model_wrapper.model.fit(train_x, train_y)

    predicted_y = model_wrapper.model.predict(test_x)

    model_wrapper.model = pipeline
    _ = model_wrapper.model.fit(total_x, total_y)
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