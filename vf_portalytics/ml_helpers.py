import logging
from xgboost import XGBRegressor
import category_encoders as ce
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

POTENTIAL_TRANSFORMER = {
    'OneHotEncoder': ce.OneHotEncoder,
    'OrdinalEncoder': ce.OrdinalEncoder,
    'TargetEncoder': ce.TargetEncoder,
    'JamesSteinEncoder': ce.JamesSteinEncoder,
    'MinMaxScaler': MinMaxScaler,
    'StandardScaler': StandardScaler
}

POTENTIAL_MODELS = {
    'XGBRegressor': XGBRegressor,
    'XGBRegressorChain': RegressorChain(XGBRegressor),
    'ExtraTreesRegressor': ExtraTreesRegressor,
}


def _initialize_model(fc_model, params):
    initialized_params = {key: value for key, value in params.items() if key in fc_model().get_params()}
    model = fc_model(**initialized_params)
    return model


def get_model(params):
    params = params if params else {}
    try:
        model_name = params.get('model_name')
        fc_model = POTENTIAL_MODELS[model_name]
        if model_name == 'XGBRegressorChain':
            # handle differently a nested model
            # no time to find the issue but when get_model is called inside MultiModel XGBRegressorChain comes
            # with initialized ExtraTreesRegressor but when get_model is called independently
            # ExtraTreesRegressor is just a module
            if callable(fc_model.base_estimator):
                fc_model.base_estimator = _initialize_model(fc_model.base_estimator, params)
            else:
                initialized_params = {'base_estimator__' + key: value for key, value in params.items()
                                      if 'base_estimator__' + key in fc_model.get_params()}
                fc_model.set_params(**initialized_params)
            fc_model.order = params.get('order')
        else:
            fc_model = _initialize_model(fc_model, params)
        return fc_model
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
        return None


class CustomTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, transformer='TargetEncoder', cols=None):
        """
        A custom transformer that supports multiple targets by using only the first target as input in the selected
        transformer and subset of features

        Parameters
        ----------
        transformer (object): a transformer from POTENTIAL_TRANSFORMER
        cols (list): a list of features that need to be transformed
                     (subset of a features of X; input of fit() and transform)
        """
        self.transformer = get_transformer(transformer)
        self.cols = cols if cols else []

    def fit(self, X, y=None):

        if self.transformer is None:
            logger.warning('transformer was not assigned in CustomTransformer, before fit')
            return self

        X = X[self.cols] if self.cols else X
        # if target is multicolumn keep only the first one for fiting the transformer
        y = y if y is None or isinstance(y, pd.Series) else y.iloc[:, 0]

        self.transformer.fit(X, y)
        return self

    def transform(self, X):

        if self.transformer is None:
            logger.warning(
                'transformer was not assigned in CustomTransformer, before transform (output identical with input)')
            return X

        if self.cols:
            return self._transform_col_subset(X)

        transformed_x = self.transformer.transform(X)
        transformed_x = self._post_process(transformed_x, X.index)
        return transformed_x

    def _transform_col_subset(self, X):
        subset_x = X[self.cols]
        subset_x = self.transformer.transform(subset_x)
        subset_x = self._post_process(subset_x, X.index)

        X.drop(self.cols, inplace=True, axis=1)
        return pd.concat([X, subset_x], axis=1)

    def _post_process(self, x, index):
        """
        sklearn scalers return numpy and category_encoders return dataframe
        """
        return x if isinstance(x, pd.DataFrame) else pd.DataFrame(x, columns=self.cols, index=index)


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


class CustomClusterTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer creates 'cluster' column that can be used as cluster field.

    """

    def __init__(self, cat_feature=None):
        """
        :param cat_feature: None or existing field name in the dataset
        """
        self.cat_feature = cat_feature

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Creates 'cluster' column for X.
        :param X: DataFrame to transform.
        :returns: X: DataFrame input with the new 'cluster' column.
        """
        if 'cluster' in X.columns:
            return X
        elif self.cat_feature == 'vf_category' or self.cat_feature is None:
            # vf_category means that no category defined
            X['cluster'] = 0.0
        else:
            try:
                X['cluster'] = X[self.cat_feature]
            except KeyError:
                raise KeyError('Feature "{}" not in dataframe'.format(self.cat_feature))
        return X
