import pandas as pd
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin


class FeatureSubsetModel(BaseEstimator, RegressorMixin):

    def __init__(self, group_cols=None, sub_models=None):
        """
        Build regression model for subsets of feature rows matching particular combination of feature columns.
        """

        self.group_cols = deepcopy(group_cols) if group_cols else None
        self.sub_models = deepcopy(sub_models) if sub_models else None

    def fit(self, X, y):
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

            # Fit the sub-model with subset of rows
            gp_model = gp_model.fit(X=x_in.values, y=y_in.values)
            self.sub_models[gp_key] = gp_model

        return self

    def predict(self, X):
        """
        Same as ``self.fit()``, but call the ``predict()`` method for each sub-model and return the results.
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