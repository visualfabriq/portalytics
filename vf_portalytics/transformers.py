from models.tool import set_categorical_features

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Replaces categorical features with binary columns for each unique category
    """
    def __init__(self, categorical_features=None):
        self.maps = dict()
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        # find automatically categorical features if they are not previously declared
        if self.categorical_features is None:
            self.categorical_features = set_categorical_features(data=X, potential_cat_feat=None)

        # Check columns are in X
        for col in self.categorical_features:
            if col not in X:
                raise ValueError('Missing categorical feature: ' + col)

        # Store each unique value for each cat feature
        for col in self.categorical_features:
            uniques = X[col].unique()
            self.maps[col] = [unique for unique in uniques]

        return self

    def transform(self, X, y=None):

        output = X.copy()
        for col, vals in self.maps.items():
            # apply label encoding
            lookup = {x: i for i, x in enumerate(vals)}
            missing = len(lookup)

            output.loc[:, col] = output.loc[:, col].apply(lambda x: lookup.get(x, missing))
            dummy_col = pd.get_dummies(X[col], prefix=col)
            # Get missing columns
            dummy_col = OneHotEncoder.get_missing_cols(dummy_col, col, lookup)

            # drop the new category (the ones that were not in any category before)
            # or even if in train set there is NaN delete it
            dummy_col = dummy_col.drop(col + '_{}'.format(len(lookup)), axis=1, errors='ignore')
            output = pd.concat([output, dummy_col], axis=1)

            del output[col]
        output = output.sort_index(axis=1)
        return output

    @staticmethod
    def get_missing_cols(dummy_col, col, lookup):
        # Get missing columns that were in the trainset but not in test
        train_columns = [col + '_{}'.format(i) for i in lookup.keys()]
        missing_cols = set(train_columns) - set(dummy_col.columns)
        for c in missing_cols:
            dummy_col[c] = 0
        dummy_col = dummy_col[train_columns]
        return dummy_col


class LabelEncoder(BaseEstimator, TransformerMixin):
    """
    Replaces categorical features with integers for each unique category
    """

    def __init__(self, categorical_features=None):
        self.maps = dict()
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        # find automatically categorical features if they are not previously declared
        if self.categorical_features is None:
            self.categorical_features = set_categorical_features(data=X, potential_cat_feat=None)

        # Check columns are in X
        for col in self.categorical_features:
            if col not in X:
                raise ValueError('Missing categorical feature: ' + col)

        for col in self.categorical_features:
            self.maps[col] = dict(zip(
                X[col].values,
                X[col].astype('category').cat.codes.values
            ))
        return self

    def transform(self, X, y=None):
        output = X.copy()
        for col, value in self.maps.items():
            # Map the column
            output[col] = output[col].map(value)

        return output

potential_transformers = {
    'OneHotEncoder': OneHotEncoder(),
    'LabelEncoder': LabelEncoder()
                          }