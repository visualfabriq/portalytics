import json
import os
import os.path

import numpy as np
import pandas as pd
from sklearn.externals import joblib

from vf_portalytics.tool import rm_file_or_dir


def _label_safe_value(input_val):
    return str(input_val).replace('.', '')


def _label_check(input_val, labels):
    if input_val != input_val:
        return -1
    else:
        return labels.get(_label_safe_value(input_val), -1)


class PredictionModel(object):
    def __init__(self, id, path=None, one_hot_encode=None):

        if id is None:
            raise ValueError('Model Id cannot be None')

        self.id = id

        # filepaths
        self.path = path or '/srv/model'
        self.meta_path = os.path.join(self.path, str(self.id) + '.meta')
        self.model_path = os.path.join(self.path, str(self.id) + '.pkl')

        # try to load metadata
        self._load_metadata()
        if one_hot_encode is not None:
            self.one_hot_encode = one_hot_encode

        # try to load model
        self._load_model()

    def _load_metadata(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as meta_file:
                model_data = json.load(meta_file)
        else:
            model_data = {}

        self.target = model_data.get('target', None)
        self.features = model_data.get('features', {})
        self.labels = model_data.get('labels', {})
        self.one_hot_encode = model_data.get('one_hot_encode', True)
        self.ordered_column_list = model_data.get('ordered_column_list', [])

    def _save_metadata(self):
        model_data = {'features': self.features,
                      'target': self.target,
                      'labels': self.labels,
                      'one_hot_encode': self.one_hot_encode,
                      'ordered_column_list': self.ordered_column_list}
        with open(self.meta_path, 'w') as meta_file:
            json.dump(model_data, meta_file)

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = None

    def _save_model(self, compress_level=3):
        if self.model is not None:
            joblib.dump(self.model, self.model_path, compress=compress_level)

    def save(self, compress_level=3):
        self._save_metadata()
        self._save_model(compress_level=compress_level)

    def create_train_df(self, df):
        return self.pre_processing(df, train_mode=True)

    def create_test_df(self, df):
        return self.pre_processing(df, train_mode=False)

    def predict(self, df):
        if self.model is None:
            raise ValueError('No model is available for prediction')

        df = self.pre_processing(df)

        prediction_df = self.model.predict(df)
        prediction_df = self._post_processing(prediction_df)

        return prediction_df

    def pre_processing(self, df, train_mode=False, silent_mode=False):
        # check the columns against the features and targets
        if not self.features:
            raise ValueError('No features defined')

        col_list = sorted(list(self.features.keys()))
        if train_mode:
            missing_columns = [x for x in self.features.keys() if x not in df]
            if missing_columns:
                raise KeyError('Missing features columns ' + ', '.join(missing_columns))

        df = df[[x for x in col_list if x in df]].copy()

        # handle categorical features
        categorical_features = [col for col, transformations in self.features.items() if 'C' in transformations]

        if categorical_features:
            if train_mode:
                # reset labels
                self.labels = {}

            output = [df]

            for col in categorical_features:
                if train_mode:
                    # refresh label encoding for this column
                    self.labels[col] = list(df[col].value_counts().index)  # sorted label based on counts
                    if not silent_mode and len(self.labels[col]) <= 2:
                        print('Column ' + col + ' has less then expected unique values for a categorical ' +
                              'feature (' + ', '.join([str(x) for x in self.labels[col]]) + ')')
                elif col not in df:
                    # skip it if we're not training and it's missing
                    if not silent_mode:
                        print('Missing categorical feature: ' + col)
                    continue

                # apply label encoding
                lookup = {x: i for i, x in enumerate(self.labels[col])}
                missing = len(lookup)
                df[col] = df[col].apply(lambda x: lookup.get(x, missing))

                if self.one_hot_encode:
                    # apply one hot encoding
                    output.append(pd.get_dummies(df[col], prefix=col))
                    del df[col]

            if self.one_hot_encode:
                df = pd.concat(output, axis=1)

        # transform columns
        for column, transforms in self.features.items():
            if not transforms:
                continue
            for transform in transforms:
                if transform == 'log':
                    df[column] = np.log(df[column])
                elif transform != 'C':
                    raise KeyError('Unknown transform option: ' + transform)

        # save the columns and order of columns
        if train_mode:
            col_list = sorted([x for x in df.columns if x not in self.target])
            self.ordered_column_list = col_list
        else:
            col_list = self.ordered_column_list

        if train_mode and not col_list:
                raise ValueError('Model does not contains enough proper features')

        # check the availability of all columns
        for col in col_list:
            if col not in df:
                if self.one_hot_encode and \
                        any(1 for feature_col in categorical_features if col.startswith(feature_col)):
                    # features (one hot encoded) should be set to 0
                    df[col] = 0
                elif col in categorical_features:
                    # features (not one hot encoded) should be set to -1
                    df[col] = -1
                else:
                    df[col] = 0.0

        # set the columns in order
        df = df[col_list]

        return df

    def _post_processing(self, ser):
        # todo: we currently do not support log/exp if we have multiple target columns
        for col, transforms in self.target.items():
            for transform in transforms:
                ser = self._back_transformation(transform, ser)
        return ser

    @staticmethod
    def _back_transformation(transform, input_ser):
        if transform == 'log':
            return np.exp(input_ser)

    def delete(self):
        rm_file_or_dir(self.meta_path)
        rm_file_or_dir(self.model_path)
