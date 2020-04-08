import pandas as pd
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.dummy import DummyClassifier

class FeatureSubsetTransform(BaseEstimator, TransformerMixin):

    def __init__(self, group_cols=None, transformer=None):
        """Build a feature tranformer"""
        self.transformer = transformer
        self.group_cols = group_cols

    def fit(self, X, y=None):
        """Drop the columns that are being used to group the data and fit the transformer"""
        x_in = X.drop([n for n in self.group_cols], axis=1)
        self.transformer = self.transformer.fit(X=x_in[['promoted_price']])
        return self

    def transform(self, X):
        x_in = X.drop([n for n in self.group_cols], axis=1)
        # transform the promoted_price collumn
        transformed_price = self.transformer.transform(X=x_in[['promoted_price']])
        # convert data into initial format
        transformed_price = pd.DataFrame(data=transformed_price, index=x_in.index,
                                         columns=self.transformer.get_feature_names(['promoted_price']))
        transformed_price.drop(['1', 'promoted_price'], axis=1, inplace=True)
        transformed_x = pd.concat([x_in, transformed_price], axis=1)
        transformed_x[list(self.group_cols)] = X[list(self.group_cols)]
        return transformed_x


class FeatureSubsetModel(BaseEstimator, RegressorMixin):

    def __init__(self, lookup_dict=None, group_cols=None, sub_models=None):
        """
        Build regression model for subsets of feature rows matching particular combination of feature columns.
        """
        self.lookup_dict = lookup_dict
        self.group_cols = group_cols
        self.sub_models = sub_models

    def fit(self, X, y=None):
        """
        Partition the training data, X, into groups for each unique combination of values in
        'self.group_cols' columns. For each group, train the appropriate model specified in 'self.sub_models'.

        If there is no sub_model for a group, predict 0
        The models are being trained using only the features that their names starts with 'promoted_price'
        """

        groups = X.groupby(by=list(self.group_cols))
        for gp_key, x_group in groups:
            # Find the sub-model for this group key
            gp_model = self.sub_models.get(gp_key, DummyClassifier(constant=0))

            # Drop the feature values for the group columns, since these are same for all rows
            # and so don't contribute anything into the prediction.
            x_in = x_group.drop([n for n in self.group_cols], axis=1)
            y_in = y.loc[x_in.index]

            # Fit the submodel with subset of rows and only collumns related to price
            gp_model = gp_model.fit(X=x_in[[col for col in x_in if col.startswith('promoted_price')]], y=y_in.values)
            self.sub_models[gp_key] = gp_model
        return self

    def predict(self, X, y=None):
        """
        Same as 'self.fit()', but call the 'predict()' method for each submodel and return the results.
        :return: predicted_market_share*predicted_market_volume*consumer_length/product_volume_per_sku
            where predicted_market_share are the outputs of the trained models
        """
        # create a new collumn by checking the lookup_dict
        X['predicted_market_volume'] = [self.lookup_dict.get((week, pr), 0)
                                        for week, pr in zip(X['yearweek'], X['original_product_dimension_44'])]
        groups = X.groupby(by=list(self.group_cols))
        results = []

        for gp_key, x_group in groups:
            x_in = x_group.drop([n for n in self.group_cols], axis=1)
            gp_model = self.sub_models.get(gp_key, DummyClassifier(constant=0).fit(x_in, [0]*len(x_in)))

            # predict market share only using price related data
            predicted_market_share = gp_model.predict(X=x_in[[col for col in x_in if col.startswith('promoted_price')]])
            predicted_market_share = pd.Series(index=x_in.index, data=predicted_market_share)

            result = predicted_market_share.mul(
                x_group['predicted_market_volume']).mul(
                x_group['consumer_length']).div(
                x_group['product_volume_per_sku']).clip(lower=0)

            results.append(result)
        return pd.concat(results, axis=0)
