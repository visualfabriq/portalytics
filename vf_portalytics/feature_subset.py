import pandas as pd
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin


class FeatureSubsetTransform(BaseEstimator, TransformerMixin):

    def __init__(self, group_cols=None, transformer=None):
        """Build a feature tranformer"""
        self.transformer = transformer
        self.group_cols = group_cols

    def fit(self, X, y=None):
        """Drop the columns that are being used to group the data and fit the transformer"""
        x_in = X.drop([n for n in self.group_cols], axis=1)
        self.transformer = self.transformer.fit(X=x_in)
        return self

    def transform(self, X):
        x_in = X.drop([n for n in self.group_cols], axis=1)
        # transform the data
        transformed_x = self.transformer.transform(X=x_in)
        # convert data into initial format
        transformed_x = pd.DataFrame(data=transformed_x, index=x_in.index,
                                     columns=self.transformer.get_feature_names(x_in.columns))
        transformed_x[list(self.group_cols)] = X[list(self.group_cols)]
        return transformed_x


class FeatureSubsetModel(BaseEstimator, RegressorMixin):

    def __init__(self, group_cols=None, sub_models=None):
        """
        Build regression model for subsets of feature rows matching particular combination of feature columns.

        """

        self.group_cols = group_cols
        self.sub_models = sub_models

    def fit(self, X, y=None):
        """
        Partition the training data, X, into groups for each unique combination of values in
        ``self.group_cols`` columns. For each group, train the appropriate model specified in
        ``self.sub_models``.
        """
        groups = X.groupby(by=list(self.group_cols))

        for gp_key, x_group in groups:
            # Find the sub-model for this group key
            gp_model = self.sub_models[gp_key]

            # Drop the feature values for the group columns, since these are same for all rows
            # and so don't contribute anything into the prediction.
            x_in = x_group.drop([n for n in self.group_cols], axis=1)
            y_in = y.loc[x_in.index]

            # Fit the submodel with subset of rows
            gp_model = gp_model.fit(X=x_in.values, y=y_in.values)
            self.sub_models[gp_key] = gp_model
        return self

    def predict(self, X, y=None):
        """
        Same as ``self.fit()``, but call the ``predict()`` method for each submodel and return the results.
        """
        groups = X.groupby(by=list(self.group_cols))
        results = []

        for gp_key, x_group in groups:
            gp_model = self.sub_models[gp_key]
            x_in = x_group.drop([n for n in self.group_cols], axis=1)

            result = gp_model.predict(X=x_in.values)

            result = pd.Series(index=x_in.index, data=result)
            results.append(result)

        return pd.concat(results, axis=0)