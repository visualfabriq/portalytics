import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import RegressorChain
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from xgboost import XGBRegressor

from vf_portalytics.ml_helpers import get_model


def test_get_model():
    # test xgboost
    params = {
        'max_depth': 4,
        'n_estimators': 150,
        'reg_alpha': 0.8,
        'model_name': 'XGBRegressor'
    }
    model = get_model(params)

    assert isinstance(model, XGBRegressor)
    assert model.max_depth == params['max_depth']
    assert model.n_estimators == params['n_estimators']
    assert model.reg_alpha == params['reg_alpha']

    # test ExtraTreesRegressor
    params = {
        'max_depth': 4,
        'n_estimators': 150,
        'min_samples_split': 20,
        'model_name': 'ExtraTreesRegressor'
    }
    model = get_model(params)

    assert isinstance(model, ExtraTreesRegressor)
    assert model.max_depth == params['max_depth']
    assert model.n_estimators == params['n_estimators']
    assert model.min_samples_split == params['min_samples_split']

    # test XGBRegressorChain
    params = {
        'max_depth': 4,
        'n_estimators': 150,
        'model_name': 'XGBRegressorChain',
        'order': [0, 1, 2, 3]
    }
    model = get_model(params)

    assert isinstance(model, RegressorChain)
    assert model.base_estimator.max_depth == params['max_depth']
    assert model.base_estimator.n_estimators == params['n_estimators']
    assert model.order == params['order']

    # test KerasRegressor
    params = {
        'input_nodes': 20,
        'nr_nodes_0': 30,
        'activation_0': 'sigmoid',
        'loss': 'mean_squared_error',
        'kernel_initializer_0': 'he_normal',
        'kernel_initializer_1': 'normal',
        'dropout': 0.20,
        'output_nodes': 3,
        'optimizer': 'adam',
        'model_name': 'KerasRegressor'
    }
    model = get_model(params)

    assert isinstance(model, KerasRegressor)
    assert isinstance(model.build_fn, Sequential)
    assert model.build_fn.input_shape == (None, 20)
    assert model.build_fn.loss == 'mean_squared_error'


