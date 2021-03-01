import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class LightWeightClusterModel(BaseEstimator, RegressorMixin):

    def __init__(self,
                 multiplication_dicts=None,
                 clustering_keys=None,
                 input_columns=None,
                 multiplication_columns=None,
                 division_columns=None,
                 sub_models=None,
                 min_observations_per_cluster=5,
                 price_column="promoted_price",
                 base_price_column="base_price",
                 baseline_column="baseline_units"):

        """
        A lightweight cluster model is less flexible than a ClusterModel, but much faster.
        For every sub_model in sub_models, you can specify the type of model. Currently, we support:
        LinearRegression and PriceElasticityModel.

        Parameters
        ----------
        multiplication_dicts : dict | None
            A dictionary of dictionaries to specify the amount each row needs to be multiplied by during prediction, and
            divided during training. These can be used to model trends and/or seasonality for example. The key of the
            first level shall contain the name of the columns to group by, and the key of the second level the values.
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
            dict contains the clustering_key, the second level the clustering value as the key and the model description
            as the value.
        min_observations_per_cluster : int | 5
            The minimum number of observations before a cluster is created. Otherwise the observations are passed on to
            the next clustering key level.
        price_column : str | "promoted_price"
            The name of the column used for the consumer promo price.
        base_price_column : str | "base_price"
            The name of the column used for the base price.
        baseline_column : str | "baseline_units"
            The name of the column used for the baseline units.
        """
        self.multiplication_dicts = multiplication_dicts if multiplication_dicts else dict()
        self.clustering_key_column_sets = clustering_keys
        self.input_columns = input_columns
        self.multiplication_columns = multiplication_columns if multiplication_columns else []
        self.division_columns = division_columns if division_columns else []
        self.sub_models = sub_models
        self.min_observations_per_cluster = min_observations_per_cluster
        self.nr_target_variables = 1
        self.price_column = price_column
        self.base_price_column = base_price_column
        self.baseline_column = baseline_column

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

        # During training, we switch multiplication and division around,
        # so we multiply by the division_columns
        for div_col in self.division_columns:
            y = y.mul(X[div_col], axis=0)

        # During training, we switch multiplication and division around,
        # so we divide by the multiplication_columns
        for mul_col in self.multiplication_columns:
            y = y.div(X[mul_col], axis=0)

        for clustering_key_column_set in self.clustering_key_column_sets:
            # We only take records into account that have not previously been used in a model yet
            clusters = X[~X.index.isin(fitted_indices)].groupby(by=list(clustering_key_column_set))

            for clustering_key, x_in in clusters:
                if len(x_in) < self.min_observations_per_cluster:
                    continue

                if "Unknown" in clustering_key:
                    continue

                # Find the sub-model for this group key
                if isinstance(self.sub_models[clustering_key_column_set], defaultdict):
                    cluster_model = self.sub_models[clustering_key_column_set][clustering_key]
                else:
                    cluster_model = self.sub_models[clustering_key_column_set].get(clustering_key, None)

                if cluster_model is None:
                    continue

                y_in = y.loc[x_in.index]

                for clustering_columns, multiplication_dict in self.multiplication_dicts.items():
                    for multiplication_key, values in x_in.groupby(list(clustering_columns)):
                        # During training, we divide by the multiplication_dict values
                        y_in.loc[values.index] /= multiplication_dict[multiplication_key]

                if not isinstance(cluster_model, LinearRegression):
                    raise AssertionError(
                        "Unsupported model type for LightWeightClusterModel: {} for ".format(type(cluster_model)) +
                        "clustering key {}. Use ClusterModel instead, or switch the sub-model ".format(clustering_key) +
                        "to a LinearRegression model.")

                # Fit the sub-model with subset of rows
                try:
                    cluster_model = cluster_model.fit(X=x_in[self.input_columns], y=y_in.values)
                    cluster_model = {"coef": cluster_model.coef_, "intercept": cluster_model.intercept_}

                    self.sub_models[clustering_key_column_set][clustering_key] = cluster_model

                    fitted_indices.append(pd.Series(x_in.index))
                except Exception as e:
                    print("Unable to train model for clustering key {}. Check if your data ".format(clustering_key) +
                          "contains zeroes in the multiplication columns for example:", e)

        return self

    def predict(self, X):
        scored_record_indices = set()
        results = []

        for clustering_key_column_set in self.clustering_key_column_sets:
            # We only score records that have not been scored yet.
            clusters = X[~X.index.isin(scored_record_indices)].groupby(by=list(clustering_key_column_set))

            for clustering_key, x_in in clusters:
                if "Unknown" in clustering_key:
                    continue

                if clustering_key in self.sub_models[clustering_key_column_set]:
                    # Do not change this to .get() with a default value: this breaks functionality for defaultdicts
                    sub_model = self.sub_models[clustering_key_column_set][clustering_key]
                else:
                    # No model found, try next clustering approach
                    continue

                if "coef" in sub_model:
                    predictions = pd.Series(LightWeightClusterModel.predict_lw_regression_model(
                        sub_model, x_in[self.input_columns]), index=x_in.index)
                elif "price_elasticity_coef" in sub_model:
                    predictions = self.predict_price_elasticity_model(sub_model, x_in)
                else:
                    raise AssertionError("Unsupported model: {}".format(sub_model))

                scored_record_indices |= set(x_in.index)
                results.append(pd.DataFrame(predictions, index=x_in.index).clip(lower=0))

        if len(results) > 0:
            results = pd.concat(results, axis=0).loc[X.index]
        else:
            results = pd.DataFrame(columns=[i for i in range(self.nr_target_variables)])

        for clustering_columns, multiplication_dict in self.multiplication_dicts.items():
            results *= X.loc[results.index][list(clustering_columns)].apply(lambda x:
                                                                            multiplication_dict.get(tuple(x), 1),
                                                                            axis=1).values.reshape(-1, 1)

        # Multiply by multiplication columns
        for mul_col in self.multiplication_columns:
            results = results.mul(X[mul_col], axis=0)

        # Divide by multiplication columns
        for div_col in self.division_columns:
            results = results.div(X[div_col], axis=0)

        # Predict -1 for the rest
        indices_to_score = X[~X.index.isin(scored_record_indices)].index
        results.loc[indices_to_score] = -1

        return results

    @staticmethod
    def predict_lw_regression_model(model, selected_columns):
        return (model["coef"] * selected_columns + model["intercept"]).values.reshape(-1)

    def predict_price_elasticity_model(self, model, X):
        # Only score valid records (i.e. records that have a base price > 0)
        valid_indices = X[self.base_price_column] > 0
        relative_prices = np.array((X[valid_indices][self.price_column] /
                                    X[valid_indices][self.base_price_column])).reshape(-1, 1)

        uplift = (model["price_elasticity_coef"] * relative_prices + model["price_elasticity_intercept"]).reshape(-1)
        result = (uplift * X[valid_indices][self.baseline_column]).clip(lower=0)

        # Add the invalid indices to create a complete set again
        result = result.append(pd.Series(np.nan, index=X[~valid_indices].index))[X.index]
        return result
