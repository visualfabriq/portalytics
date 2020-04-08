import pandas as pd
import xgboost
import copy

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.dummy import DummyClassifier

from vf_portalytics.tool import set_categorical_features
from vf_portalytics.transformers import OneHotEncoder, potential_transformers

class Grouped_Model(BaseEstimator, RegressorMixin):

    def __init__(self, group_col=None, clusters=None, params=None, selected_features=None):
        """
        Build a model for each subset of rows matching particular category of features group_col.
        Input:
            group_col: string; name of the column that the groups exist
            clusters: array; with the name of unique groups
            params: dictionary with keys the group names and values dictionaries of the selected hyperparameters
        """
        self.group_col = group_col
        self.clusters = clusters
        self.params = params
        self.selected_features = selected_features
        self.categorical_features = dict()
        self.sub_models, self.transformers = self.initiliaze_models()

    def fit(self, X, y=None):
        """
        Partition the training data, X, into groups for each unique combination of values in
        'self.group_col' columns. For each group, train the appropriate model specified in 'self.sub_models'.
        If there is no sub_model for a group, predict 0
        The models are being trained using only the features that their names starts with 'promoted_price'
        """
        groups = X.groupby(by=self.group_col)
        for gp_key, group in groups:
            x_group = group[self.selected_features[gp_key]]
            y_in = y.loc[x_group.index]
            # preprocessing
            self.categorical_features[gp_key] = set_categorical_features(data=x_group,
                                                                         potential_cat_feat=self.params[gp_key].get(
                                                                             'potential_cat_feat', None))
            gp_transformer = self.transformers.get(gp_key)
            gp_transformer.categorical_features = self.categorical_features[gp_key]
            gp_transformer.fit(x_group)
            x_group = gp_transformer.transform(x_group)

            # Find the sub-model for this group key and fit
            gp_model = self.sub_models.get(gp_key, DummyClassifier(constant=0))
            gp_model = gp_model.fit(X=x_group, y=y_in.values)

            self.sub_models[gp_key] = gp_model
            self.transformers[gp_key] = gp_transformer
            print('Model for ' + gp_key + ' trained')
        return self

    def predict(self, X, y=None):
        """
        Same as 'self.fit()', but call the 'predict()' method for each submodel and return the results.
        """
        groups = X.groupby(by=self.group_col)
        results = []
        for gp_key, group in groups:
            x_group = group[self.selected_features[gp_key]]
            gp_transformer = self.transformers[gp_key]
            gp_model = self.sub_models.get(gp_key, DummyClassifier(constant=0).fit(x_group, [0] * len(x_group)))
            # preprocessing
            x_group = gp_transformer.transform(x_group)
            # predict
            result = gp_model.predict(x_group)
            result = pd.Series(index=x_group.index, data=result)
            results.append(result)
        results = pd.concat(results, axis=0)
        results = pd.Series(index=X.index, data=results, name='predicted')
        return results

    def initiliaze_models(self):
        sub_models = {}
        transformers = {}
        for gp_key in self.clusters:
            sub_models[gp_key] = xgboost.XGBRegressor(
                n_estimators=self.params[gp_key].get('n_estimators', 100),
                max_depth=self.params[gp_key].get('max_depth', 3),
                subsample=self.params[gp_key].get('subsample', 1),
                min_child_weight=self.params[gp_key].get('min_child_weight', 1),
                gamma=self.params[gp_key].get('gamma', 0),
                colsample_bytree=self.params[gp_key].get('colsample_bytree', 1),
                learning_rate=self.params[gp_key].get('learning_rate', 0.1),
                silent=True
            )
            transformer_name = self.params[gp_key].get('transformer', 'OneHotEncoder')
            transformers.update({gp_key: copy.deepcopy(transformers.potential_transformers).get(transformer_name)})
        return sub_models, transformers
