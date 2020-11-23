import logging
import xgboost
import category_encoders as ce
import pandas as pd
from sklearn import ensemble
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

POTENTIAL_TRANSFORMER = {
    'OneHotEncoder': ce.OneHotEncoder,
    'OrdinalEncoder': ce.OrdinalEncoder,
    'TargetEncoder': ce.TargetEncoder,
    'JamesSteinEncoder': ce.JamesSteinEncoder
}

POTENTIAL_MODELS = {
    'XGBRegressor': xgboost.XGBRegressor,
    'ExtraTreesRegressor': ensemble.ExtraTreesRegressor,
}


def get_model(params):
    params = params if params else {}
    try:
        model_name = params.get('model_name')
        fc_model = POTENTIAL_MODELS[model_name]
        initialized_params = {key: value for key, value in params.items()
                              if key in fc_model._get_param_names()}
        model = fc_model(**initialized_params)
        return model
    except KeyError:
        logger.exception("KeyError: The '%s ' is not a currently supported model. "
                         "XGBRegressor is being used" % str(model_name))
        fc_model = POTENTIAL_MODELS['XGBRegressor']
        model = fc_model()
        return model


def get_transformer(name):
    try:
        output = POTENTIAL_TRANSFORMER[name]
        return output()
    except KeyError:
        logger.exception("KeyError: The '%s' is not a potential transformer. TargetEncoder is being used" % str(name))
        return ce.TargetEncoder()


class CustomTransformer(BaseEstimator, TransformerMixin):
    """A custom transformer that supports multiple targets by using only the first target as input in the selected
    transformer. """

    def __init__(self, transformer='TargetEncoder'):
        self.transformer = get_transformer(transformer)

    def fit(self, X, y=None):
        if y is None or isinstance(y, pd.Series):
            self.transformer.fit(X, y)
        else:
            self.transformer.fit(X, y.iloc[:, 0])
        return self

    def transform(self, X, y=None):
        if y is None or isinstance(y, pd.Series):
            return self.transformer.transform(X, y)
        else:
            return self.transformer.transform(X, y.iloc[:, 0])
