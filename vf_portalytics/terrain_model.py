import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import vf_portalytics.terrain_config as config


def _get_dataframe(file_format, obj, key_cols=None):
    """Check whether obj is dataframe or string and read in if necessary."""
    if isinstance(obj, str):
        obj = getattr(pd, f"read_{file_format}")(obj)
    elif not isinstance(obj, pd.DataFrame):
        raise ValueError("obj must be a path (string) or dataframe.")

    if key_cols is None:
        return obj

    return obj.sort_values(key_cols).set_index(key_cols)


class Terrain(BaseEstimator):
    """Provides portalytics interface to the Terrain model."""

    def __init__(
        self,
        account_id_col=config.account_id_col,
        account_id_mapper_path=config.account_id_mapper_path,
        baseline_col=config.baseline_col,
        coefs_path=config.coefs_path,
        customer_col=config.customer_col,
        file_format=config.data_file_format,
        item_col=config.item_col,
        model_transforms=config.transforms,
        pid_mapper_path=config.pid_mapper_path,
        prod_line_col=config.prod_line_col,
        promo_coefs=config.promo_coefs,
        promo_features=config.promo_features,
    ):
        """Initialize Terrain model.

        Args:
            account_id_col (str, optional): Name of account id column in VF call data.
            account_id_mapper_path (str, optional): Path to account id mapper. Defaults
                to config.account_id_mapper_path.
            baseline_col (str, optional): Name of baseline column in VF call data.
                Defaults to config.baseline_col.
            coefs_path (str, optional): Path to coefficient data. Defaults to
                config.coefs_path.
            customer_col (str, optional): Name of customer column in coef table.
                Defaults to config.customer_col.
            file_format (str, optional): File format of baseline data. Defaults to
                config.data_file_format. Should be one of "parquet", "csv", or "json".
            item_col (str, optional): Name of item column in VF call data. Defaults to
                config.item_col.
            model_transforms (dict, optional): Dictionary of model transforms to be used
                in predict method. Defaults to config.transforms.
            pid_mapper_path (str, optional): Path to product id mapper. Defaults to
                config.pid_mapper_path.
            prod_line_col (str, optional): Name of product line column in coef table.
            promo_coefs (list, optional): Names of promo coefficients. Defaults to
                config.promo_coefs.
            promo_features (list, optional): List of promo features to be used in VF.
                Defaults to config.promo_features.
        """
        if file_format not in ["parquet", "csv", "json"]:
            raise ValueError("file_format must be one of 'parquet', 'csv', or 'json'")
        self.coef_table = _get_dataframe(file_format, coefs_path)
        self.account_id_mapper = _get_dataframe(file_format, account_id_mapper_path)
        self.pid_mapper = _get_dataframe(file_format, pid_mapper_path)
        self.account_id_col = account_id_col
        self.baseline_col = baseline_col
        self.customer_col = customer_col
        self.item_col = item_col
        self.prod_line_col = prod_line_col
        self.model_transforms = model_transforms
        self.promo_coefs = promo_coefs
        self.promo_features = promo_features

    def fit(self, X, y=None):
        """Fit Terrain model.

        Args:
            X (Union[pd.DataFrame, np.array]): Data to fit model on.
            y (Union[pd.Series, np.array], optional): Target data. Defaults to None.

        Returns:
            Terrain: Fitted Terrain model.

        Note: This is a dummy method to conform to sklearn API. It does nothing. We
            will not be fitting the Terrain model inside the Portalytics framework.
        """
        return self

    def predict(self, X):
        """Predict Terrain model.

        Args:
            X (pd.DataFrame): Data to predict on.

        Returns:
            pd.DataFrame: Predicted values.
        """
        out = []
        for tup in X.itertuples(index=False):
            factor = self._get_factor(tup)
            out.append(factor * getattr(tup, self.baseline_col))

        return pd.Series(out).round()

    def _compute_factor(self, coefs, values):
        return np.prod(
            [
                self.model_transforms[feat](val, coef)
                for feat, coef, val in zip(self.promo_features, coefs, values)
            ]
        )

    def _get_factor(self, tup):
        """Return baseline scaling factor.

        Args:
            tup (pandas.DataFrame.itertuples named tuple): Row of dataframe passed to
                predict as a named tuple.

        Returns:
            float: Scaling factor to multiply against baseline.
        """
        input_values = [getattr(tup, col) for col in self.promo_features]
        try:
            prod_line = self.pid_mapper.loc[
                getattr(tup, self.item_col), self.prod_line_col
            ]

        except KeyError:
            return 1.0
        try:
            cpe_1 = self.account_id_mapper.loc[
                getattr(tup, self.account_id_col), self.customer_col
            ]
        except KeyError:
            cpe_1 = "SIMPLIFIED"
        try:
            coefs = self.coef_table.loc[(cpe_1, prod_line)]
        except KeyError:
            try:
                coefs = self.coef_table.loc[("SIMPLIFIED", prod_line)]
            except KeyError:
                return 1.0
        return self._compute_factor(coefs, input_values)
