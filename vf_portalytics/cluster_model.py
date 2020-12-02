import logging

import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.base import BaseEstimator, RegressorMixin


class ClusterModel(BaseEstimator, RegressorMixin):

    def __init__(self,
                 seasonality_dict=None,
                 seasonality_key_columns=None,
                 clustering_keys=None,
                 input_columns=None,
                 multiplication_columns=None,
                 division_columns=None,
                 sub_models=None,
                 min_observations_per_cluster=5):
        """
        Build regression model for subsets of feature rows matching particular combination of feature columns.

        Parameters
        ----------
        seasonality_dict : dict | None
            A dictionary to specify the amount each row needs to be multiplied by during prediction, and divided during
            training.
        seasonality_key_columns : str[] | None
            A list of columns to select the right value from the seasonality_dict.
        clustering_keys : str[][]
            An ordered list of lists of column names that are to be used in the clustering. Items that are earlier in
            the list take preference over those later in the list.
        input_columns : str[]
            A list of columns in the DataFrames that are used for predict() and fit() that contain the explanatory
            variables.
        multiplication_columns : str[]
            A list of columns in the DataFrames that are used as multipliers during predict() and as divisors during
            fit() to scale the target variable with.
        division_columns : str[]
            A list of columns in the DataFrames that are used as divisors during predict() and as multipliers during
            fit() to scale the target variable with.
        sub_models : dict | None
            A dictionary of dictionaries that contain the models that are to be fitted and used. The first level of the
            dict contains the clustering_key, the second level
        min_observations_per_cluster : int | 5
            The minimum number of observations before a cluster is created. Otherwise the observations are passed on to
            the next clustering key level.
        """
        self.seasonality_dict = seasonality_dict
        self.seasonality_key_columns = seasonality_key_columns
        self.clustering_key_column_sets = clustering_keys
        self.input_columns = input_columns
        self.multiplication_columns = multiplication_columns
        self.division_columns = division_columns
        self.sub_models = sub_models
        self.min_observations_per_cluster = min_observations_per_cluster
        self.nr_target_variables = 1

        if seasonality_dict is not None and (seasonality_key_columns is None or len(seasonality_key_columns) == 0):
            raise KeyError("Using seasonality_dict without seasonality_key_columns")

        if seasonality_key_columns is not None and len(seasonality_key_columns) > 0 and seasonality_dict is None:
            raise KeyError("Using seasonality_key_columns without seasonality_dict")

    def fit(self, X, y=None):
        """
        For each clustering group, as specified by clustering_key_column_sets, split the data by the values
        in the clustering_key_column_set. If there are enough observations, train a model. If there are too few
        observations, leave these observations for the next (more generic) clustering_key_column_set, to train
        a model where multiple clusters can be combined.

        If there is no model for a group, predict -1.

        The models are being trained using only the features that are available in self.input_columns."""
        fitted_indices = pd.Series()

        if hasattr(y, "iloc"):
            if hasattr(y.iloc[0], "__len__"):
                self.nr_target_variables = len(y.iloc[0])
        elif hasattr(y[0], "__len__"):
            self.nr_target_variables = len(y[0])

        # Calculate the seasonality_values if applicable
        if self.seasonality_dict is not None:
            seasonality_values = \
                X.apply(lambda x: self.seasonality_dict.get(tuple(x[c] for c in self.seasonality_key_columns), 1),
                        axis=1)

        for clustering_key_column_set in self.clustering_key_column_sets:
            # We only take records into account that have not previously been used in a model yet
            clusters = X[~X.index.isin(fitted_indices)].groupby(by=list(clustering_key_column_set))

            for clustering_key, cluster in clusters:
                if len(cluster) < self.min_observations_per_cluster:
                    continue

                if "Unknown" in clustering_key:
                    continue

                # Find the sub-model for this group key
                if isinstance(self.sub_models[clustering_key_column_set], defaultdict):
                    cluster_model = self.sub_models[clustering_key_column_set][clustering_key]
                else:
                    cluster_model = self.sub_models[clustering_key_column_set].get(clustering_key,
                                                                                   None)

                if cluster_model is None:
                    continue

                # Drop the feature values for the group columns, since these are same for all rows
                # and so don't contribute anything into the prediction.
                x_in = cluster.drop([n for n in clustering_key_column_set], axis=1)
                y_in = y.loc[x_in.index]

                if self.seasonality_dict is not None:
                    # During training, we divide by the seasonality_values
                    y_in /= seasonality_values.loc[x_in.index].values

                # During training, we switch multiplication and division around,
                # so we multiply by the division_columns
                for div_col in self.division_columns:
                    y_in *= cluster.loc[x_in.index][div_col].values

                # During training, we switch multiplication and division around,
                # so we divide by the multiplication_columns
                for mul_col in self.multiplication_columns:
                    # During training, we switch multiplication and division around
                    y_in /= cluster.loc[x_in.index][mul_col].values

                # Fit the sub-model with subset of rows
                try:
                    cluster_model = cluster_model.fit(X=x_in[self.input_columns], y=y_in.values)
                    self.sub_models[clustering_key_column_set][clustering_key] = cluster_model

                    fitted_indices.append(pd.Series(cluster.index))
                except Exception as e:
                    print("Unable to train model for clustering key {}. Check if your data ".format(clustering_key) +
                          "contains zeroes in the multiplication columns for example:", e)
                    logging.exception("Exception")

        return self

    def predict(self, X):
        """
        Same as 'self.fit()', but call the 'predict()' method for each submodel and return the results.

        Returns
        -------
        The predicted values, multiplied by the self.multiplication_columns and divided by the self.division_columns.
        Also multiplied by the seasonality_values, if applicable.
        """
        # Prepare the seasonality values
        if self.seasonality_dict is not None:
            seasonality_values = \
                X.apply(lambda x: self.seasonality_dict.get(tuple(x[c] for c in self.seasonality_key_columns), 1),
                        axis=1)

        results = []
        scored_record_indices = set()

        for clustering_key_column_set in self.clustering_key_column_sets:
            # We only score records that have not been scored yet.
            clusters = X[~X.index.isin(scored_record_indices)].groupby(by=list(clustering_key_column_set))

            for clustering_key, cluster in clusters:
                if "Unknown" in clustering_key:
                    continue

                if clustering_key in self.sub_models[clustering_key_column_set]:
                    # Do not change this to .get() with a default value: this breaks functionality for defaultdicts
                    sub_model = self.sub_models[clustering_key_column_set][clustering_key]
                else:
                    # No model found, try next clustering approach
                    continue

                # Drop the feature values for the group columns, since these are same for all rows
                # and so don't contribute anything into the prediction.
                x_in = cluster.drop([n for n in clustering_key_column_set], axis=1)

                try:
                    predictions = sub_model.predict(x_in[self.input_columns])

                    # Multiply by seasonality
                    if self.seasonality_dict is not None:
                        predictions *= seasonality_values.loc[x_in.index].values.reshape(-1)

                    # Multiply by multiplication columns
                    for mul_col in self.multiplication_columns:
                        predictions *= x_in.loc[x_in.index][mul_col].values

                    # Divide by multiplication columns
                    for div_col in self.division_columns:
                        predictions /= x_in.loc[x_in.index][div_col].values
                except Exception:
                    # ALlow for fallback models, like the PriceElasticityModel.
                    # In that case, pass on all variables to the model, and don't
                    # do any post-processing on the results.
                    predictions = sub_model.predict(x_in)

                scored_record_indices |= set(cluster.index)
                results.append(pd.DataFrame(predictions, index=cluster.index).clip(lower=0))

        # Predict -1 for the rest
        indices_to_score = X[~X.index.isin(scored_record_indices)].index
        results.append(pd.DataFrame(-np.ones(self.nr_target_variables * len(indices_to_score))
                                       .reshape(len(indices_to_score), self.nr_target_variables),
                                    index=indices_to_score))

        return pd.concat(results, axis=0).loc[X.index]


class ClusterSubModel(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        """
        :param model: Model to set max- and min values on
        :param max_value: Maximum value the model should predict
        :param min_value: Minimum value the model should predict
        """
        self.model = model
        self.max_value = None
        self.min_value = None

    def fit(self, X, y):
        self.max_value = np.array(y).max()
        self.min_value = np.array(y).min()

        self.model.fit(X, y)

        return self

    def predict(self, X):
        return self.model.predict(X).clip(min=self.min_value, max=self.max_value)


class PriceElasticityModel(BaseEstimator, RegressorMixin):
    sub_model = None
    price_column = None
    base_price_column = None
    baseline_column = None

    def __init__(self,
                 sub_model,
                 price_column="promoted_price",
                 base_price_column="base_price",
                 baseline_column="baseline_units"):
        """
        :param sub_model: Dict of sub models
        :param price_column: Promoted price
        :param base_price_column: Base price
        :param baseline_column: Baseline units
        """
        self.sub_model = sub_model
        self.price_column = price_column
        self.base_price_column = base_price_column
        self.baseline_column = baseline_column

    def fit(self, X, y):
        y = y / X[self.baseline_column]
        X = X[self.price_column] / X[self.base_price_column]

        # Only train on records that do not cause errors
        mask = ~np.isnan(X) & np.isfinite(X) & ~np.isnan(y) & np.isfinite(y)
        self.sub_model.fit(np.array((X[mask])).reshape(-1, 1), y[mask])

        return self

    def predict(self, X):
        # Only score valid records (i.e. records that have a base price > 0)
        valid_indices = X[self.base_price_column] > 0

        result = (self.sub_model.predict(np.array((X[valid_indices][self.price_column] /
                                                   X[valid_indices][self.base_price_column])).reshape(-1, 1)) *
                  X[valid_indices][self.baseline_column]).clip(lower=0)

        result = result.append(pd.Series(np.nan, index=X[~valid_indices].index))[X.index]

        return result
