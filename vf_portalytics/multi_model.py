import logging

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.dummy import DummyClassifier

from vf_portalytics.tool import get_categorical_features
from vf_portalytics.ml_helpers import get_model, CustomTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MultiModel(BaseEstimator, RegressorMixin):

    def __init__(self, group_col=None, clusters=None, params=None,
                 selected_features=None, nominals=None, ordinals=None):
        """
        Build a model for each subset of rows matching particular category of features group_col.

        Parameters:
            group_col: string
                name of the column that the groups exist
            clusters: array
                the name of unique groups
            params: dictionary of dictionaries
                the group names with dicts that include the selected  model_name, its hyperparameters,
                transformer names to be used.
                Example
                -------
                {'A': {
                        'model_name': 'XGBRegressor',
                        'transformer_nominal': 'TargetEncoder',
                        'transformer_ordinal': 'OrdinalEncoder'
                    }
                }
            selected_features: dictionary
                keys the cluster name and values list of selected features
            nominals: list
                features (from all clusters) that are nominal
            ordinals: list
                features (from all clusters) that are ordinal
        """
        self.group_col = group_col
        self.clusters = clusters
        self.params = params
        self.selected_features = selected_features
        self.nominals = nominals
        self.ordinals = ordinals

        self.sub_models = self.initiliaze_models()
        self.multi_transformer = None

    def fit(self, X, y=None):
        """
        Partition the training data, X, into groups for each unique combination of values in
        'group_col' columns. For each group, train the appropriate model specified in 'sub_models'.

        If there is no sub_model for a group, predict 0.
        """
        # preprocessing
        self.multi_transformer = MultiTransformer(self.group_col, self.clusters,
                                                  self.selected_features, self.nominals, self.ordinals, self.params)
        transformed_x = self.multi_transformer.fit_transform(X, y)
        transformed_x[self.group_col] = X[self.group_col]
        del X

        groups = transformed_x.groupby(by=self.group_col)
        for gp_key, x_group in groups:
            x_group = x_group.drop(columns=[self.group_col])
            y_in = y.loc[x_group.index]
            # Find the sub-model for this group key
            try:
                gp_model = self.sub_models[gp_key]
            except KeyError:
                logger.exception('There was no model initialized for category %s' % str(gp_key))
                logger.exception('A Dummy Classifier was chosen')
                gp_model = DummyClassifier(constant=0)
            # fit
            gp_model = gp_model.fit(X=x_group, y=y_in.values)

            self.sub_models[gp_key] = gp_model
            print('Model for %s trained' % str(gp_key))
        return self

    def predict(self, X, y=None):

        # single model
        if self.group_col not in X.columns:
            if not len(self.clusters) == 1:
                raise AssertionError('The features that indicates categories in trainset do not exist in new data')
            X[self.group_col] = self.clusters[0]

        transformed_x = self.multi_transformer.transform(X, y)
        transformed_x[self.group_col] = X[self.group_col]
        transformed_x = transformed_x.reindex(X.index)
        del X

        groups = transformed_x.groupby(by=self.group_col)
        results = []
        for gp_key, x_group in groups:
            x_group = x_group.drop(columns=[self.group_col])

            # Find the sub-model for this group key and fit
            try:
                gp_model = self.sub_models[gp_key]
            except KeyError:
                logger.exception('There was no model initialized for category %s' % str(gp_key))
                logger.exception('A Dummy Classifier was chosen')
                gp_model = DummyClassifier(constant=0).fit(x_group, [0] * len(x_group))

            # predict
            result = gp_model.predict(x_group)
            result = pd.DataFrame(index=x_group.index, data=result)
            results.append(result)
        results = pd.concat(results, axis=0)
        results = pd.DataFrame(index=transformed_x.index, data=results, dtype=np.float64)
        return results

    def initiliaze_models(self):
        sub_models = {}
        for gp_key in self.clusters:
            sub_models[gp_key] = get_model(self.params[gp_key])
        return sub_models


class MultiTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, group_col,  clusters, selected_features, nominals, ordinals, params):
        """
        Build a transformer for each subset of rows matching particular category of features group_col.

        Parameters:
            group_col: string
                name of the column that the groups exist
            clusters: array:
                the name of unique groups
            selected_features: dictionary
                keys the cluster name and values list of selected features
            nominals: list
                features (from all clusters) that are nominal
            ordinals: list
                features (from all clusters) that are ordinal
            params: dictionary of dictionaries
                the group names with dicts that include the selected  model_name, its hyperparameters,
                transformer names to be used.
                Example
                -------
                {'A': {
                        'transformer_nominal': 'TargetEncoder',
                        'transformer_ordinal': 'OrdinalEncoder'
                    }
                }
        """
        self.group_col = group_col
        self.clusters = clusters
        self.selected_features = selected_features
        self.nominals = nominals
        self.ordinals = ordinals
        self.params = params

        self.categorical_features = dict()
        self.transformers_nominals = None
        self.transformers_ordinals = None
        self.transformers_nominals, self.transformers_ordinals = self.initiliaze_transformers()

    def fit(self, X, y=None):
        """
        Partition the training data, X, into groups for each unique combination of values in
        'group_col' columns. For each group, train the appropriate transformer specified in
        transformers_nominals or in transformers_ordinals .
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
                # its transformed because if OneHotEncoder, next transformer needs equal feature size when transforming
                x_group = gp_transformer_nominals.fit_transform(x_group, y_in)

            # preprocess ordinals
            gp_transformer_ordinals = self.transformers_ordinals.get(gp_key)
            if gp_transformer_ordinals:
                gp_ordinals = [feature for feature in self.categorical_features[gp_key] if feature in self.ordinals]
                gp_transformer_ordinals.cols = gp_ordinals
                gp_transformer_ordinals.fit(x_group, y_in)

            self.transformers_nominals[gp_key] = gp_transformer_nominals
            self.transformers_ordinals[gp_key] = gp_transformer_ordinals
            print('Transformer for %s fitted' % str(gp_key))
        return self


    def transform(self, X, y=None):

        groups = X.groupby(by=self.group_col)
        transformed_x = pd.DataFrame()

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

            transformed_x = transformed_x.append(x_group)

        return transformed_x

    def initiliaze_transformers(self):
        transformers_nominals = {}
        transformers_ordinals = {}
        for gp_key in self.clusters:

            # nominals
            transformer_name = self.params[gp_key].get('transformer_nominal')
            transformers_nominals.update({gp_key: CustomTransformer(transformer_name)})
            # ordinals
            transformer_name = self.params[gp_key].get('transformer_ordinal')
            transformers_ordinals.update({gp_key: CustomTransformer(transformer_name)})

        return transformers_nominals, transformers_ordinals