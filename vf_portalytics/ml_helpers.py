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

    """

    A custom transformer that supports multiple targets by using only the first target as input in the selected
    transformer.
    """

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


class AccountClusterTransformer(BaseEstimator, TransformerMixin):

    """

    A custom transformer that can use a list of accounts to create clusters
    """

    def __init__(self, cat_feature_list=None):
        """
        :param cat_feature_list: List with single feature name or feature values of account_banner
        """
        self.cat_feature_list = cat_feature_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Creates new cluster column for X by using cat_feature_list and mapping the rows for each category
        :param X: DataFrame to transform
        :returns: X: DataFrame input with the new 'cluster' column
        """
        if 'cluster' in X.columns:
            return X
        if 'vf_category' in self.cat_feature_list or self.cat_feature_list is None:
            # vf_category means that no category defined
            X['cluster'] = 0.0
        elif len(self.cat_feature_list) == 1:
            # Single category
            try:
                X['cluster'] = X[self.cat_feature_list[0]]
            except KeyError:
                raise KeyError('Feature "{}" not in dataframe'.format(self.cat_feature_list[0]))
        else:
            # Multiple categories (accounts)
            cluster_map = dict()
            for account in X['account_banner'].unique():
                if account in self.cat_feature_list:
                    cluster_map[account] = account
                else:
                    cluster_map[account] = 'general_cluster'
            X['cluster'] = X['account_banner'].replace(cluster_map)
        return X
