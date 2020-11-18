from functools import partial

import category_encoders as ce
import xgboost
from sklearn import ensemble

from vf_portalytics.tool import squared_error_objective_with_weighting

def XGBRegressor(params):
    model = xgboost.XGBRegressor(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 3),
        subsample=params.get('subsample', 1),
        min_child_weight=params.get('min_child_weight', 1),
        gamma=params.get('gamma', 0),
        colsample_bytree=params.get('colsample_bytree', 1),
        objective=partial(squared_error_objective_with_weighting,
                          under_predict_weight=params.get('under_predict_weight', 2.0)),
        learning_rate=params.get('learning_rate', 0.1),
        silent=True
    )
    return model


def ExtraTreesRegressor(params):
    model = ensemble.ExtraTreesRegressor(
        n_estimators=params.get('n_estimators', 100),
        criterion=params.get('criterion', 'mse'),
        max_depth=params.get('max_depth', None),
        min_samples_split=params.get('min_samples_split', 2),
        min_samples_leaf=params.get('min_samples_leaf', 1),
        min_weight_fraction_leaf=params.get('min_weight_fraction_leaf', 0.),
        max_features=params.get('max_features', "auto"),
        max_leaf_nodes=params.get('max_leaf_nodes', None),
        min_impurity_decrease=params.get('min_impurity_decrease', 0.),
        min_impurity_split=params.get('min_impurity_split', None),
        bootstrap=params.get('bootstrap', False),
        oob_score=params.get('oob_score', False),
        random_state=1234,
        n_jobs=8
    )
    return model


def get_model(name, params):
    potential_models = {
        'XGBRegressor': XGBRegressor,
        'ExtraTreesRegressor': ExtraTreesRegressor,
    }

    fc_model = potential_models.get(name)
    if fc_model:
        model = fc_model(params)
        return model
    else:
        print('KeyError: The "%s" is not a currently supported model' % str(name))
        return


