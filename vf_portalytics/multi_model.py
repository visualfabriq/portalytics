import pandas as pd
import numpy as np
import xgboost
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier
from functools import partial

from vf_portalytics.tool import get_categorical_features, squared_error_objective_with_weighting
from vf_portalytics.transformers import get_transformer


class MultiModel(BaseEstimator, RegressorMixin):

    def __init__(self, group_col=None, clusters=None, params=None,
                 selected_features=None, nominals=None, ordinals=None):
        """
        Build a model for each subset of rows matching particular category of features group_col.
        Input:
            group_col: string; name of the column that the groups exist
            clusters: array; with the name of unique groups
            params: dictionary with keys the group names and values dictionaries of the selected hyperparameters
            selected_features: a dictionary with keys the cluster name and values list of selected features
            nominals: list of the features (from all clusters) that are nominal
            ordinals: list of the features (from all clusters) that are ordinal
        """
        self.group_col = group_col
        self.clusters = clusters
        self.params = params
        self.selected_features = selected_features
        self.nominals = nominals
        self.ordinals = ordinals
        self.categorical_features = dict()
        self.sub_models, self.transformers_nominals, self.transformers_ordinals = self.initiliaze_models()

    def fit(self, X, y=None):
        """
        Partition the training data, X, into groups for each unique combination of values in
        'group_col' columns. For each group, train the appropriate model specified in 'sub_models'.
        If there is no sub_model for a group, predict 0
        The models are being trained using only the features that their names starts with 'promoted_price'
        """
        groups = X.groupby(by=self.group_col)
        for gp_key, group in groups:
            x_group = group[self.selected_features[gp_key]]
            y_in = y.loc[x_group.index]
            # preprocessing
            self.categorical_features[gp_key] = get_categorical_features(data=x_group,
                                                                         potential_cat_feat=self.params[gp_key].get(
                                                                             'potential_cat_feat', None))
            # preprocess nominals
            gp_transformer_nominals = self.transformers_nominals.get(gp_key)
            if gp_transformer_nominals:
                gp_nominals = [feature for feature in self.categorical_features[gp_key] if feature in self.nominals]
                gp_transformer_nominals.cols = gp_nominals
                x_group = gp_transformer_nominals.fit_transform(x_group, y_in)

            # preprocess ordinals
            gp_transformer_ordinals = self.transformers_ordinals.get(gp_key)
            if gp_transformer_ordinals:
                gp_ordinals = [feature for feature in self.categorical_features[gp_key] if feature in self.ordinals]
                gp_transformer_ordinals.cols = gp_ordinals
                x_group = gp_transformer_ordinals.fit_transform(x_group, y_in)

            # Find the sub-model for this group key
            try:
                gp_model = self.sub_models[gp_key]
            except KeyError:
                print('There was no model initialized for category %s' % str(gp_key))
                print('A Dummy Classifier was chosen')
                gp_model = DummyClassifier(constant=0)

            # fit
            gp_model = gp_model.fit(X=x_group, y=y_in.values)

            self.sub_models[gp_key] = gp_model
            self.transformers_nominals[gp_key] = gp_transformer_nominals
            self.transformers_ordinals[gp_key] = gp_transformer_ordinals
            print('Model for %s trained' % str(gp_key))
        return self

    def predict(self, X, y=None):
        # single model
        if self.group_col not in X.columns:
            if not len(self.clusters) == 1:
                raise AssertionError('The features that indicates categories in trainset do not exist in new data')
            X[self.group_col] = self.clusters[0]

        groups = X.groupby(by=self.group_col)
        results = []
        for gp_key, group in groups:
            x_group = group[self.selected_features.get(gp_key, group.columns)]
            # preprocessing
            # nominals
            gp_transformer_nominals = self.transformers_nominals.get(gp_key)
            if gp_transformer_nominals:
                x_group = gp_transformer_nominals.transform(x_group)
            # ordinals
            gp_transformer_ordinals = self.transformers_ordinals.get(gp_key)
            if gp_transformer_ordinals:
                x_group = gp_transformer_ordinals.transform(x_group)

            # Find the sub-model for this group key and fit
            try:
                gp_model = self.sub_models[gp_key]
            except KeyError:
                print('There was no model initialized for category %s' % str(gp_key))
                print('A Dummy Classifier was chosen')
                gp_model = DummyClassifier(constant=0).fit(x_group, [0] * len(x_group))

            # predict
            result = gp_model.predict(x_group)
            result = pd.Series(index=x_group.index, data=result)
            results.append(result)
        results = pd.concat(results, axis=0)
        results = pd.Series(index=X.index, data=results, name='predicted', dtype=np.float64)
        return results

    def initiliaze_models(self):
        sub_models = {}
        transformers_nominals = {}
        transformers_ordinals = {}
        for gp_key in self.clusters:
            sub_models[gp_key] = xgboost.XGBRegressor(
                n_estimators=self.params[gp_key].get('n_estimators', 100),
                max_depth=self.params[gp_key].get('max_depth', 3),
                subsample=self.params[gp_key].get('subsample', 1),
                min_child_weight=self.params[gp_key].get('min_child_weight', 1),
                gamma=self.params[gp_key].get('gamma', 0),
                colsample_bytree=self.params[gp_key].get('colsample_bytree', 1),
                objective=partial(squared_error_objective_with_weighting,
                                  under_predict_weight=self.params[gp_key].get('under_predict_weight', 2.0)),
                learning_rate=self.params[gp_key].get('learning_rate', 0.1),
                silent=True
            )
            # nominals
            transformer_name = self.params[gp_key].get('transformer_nominal')
            transformers_nominals.update({gp_key: get_transformer(transformer_name)})
            # ordinals
            transformer_name = self.params[gp_key].get('transformer_ordinal')
            transformers_ordinals.update({gp_key: get_transformer(transformer_name)})

        return sub_models, transformers_nominals, transformers_ordinals
