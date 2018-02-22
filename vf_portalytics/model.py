from sklearn.externals import joblib
import numpy as np
import os.path
import os
from vf_portalytics.tool import rm_file_or_dir
import json
import gc


def _label_safe_value(input_val):
    return str(input_val).replace('.', '')


def _label_check(input_val, labels):
    if input_val != input_val:
        return -1
    else:
        return labels.get(_label_safe_value(input_val), -1)


class PredictionModel(object):
    def __init__(self, id, path=None):

        if id is None:
            raise ValueError('Model Id cannot be None')

        self.id = id

        # filepaths
        self.path = path or '/srv/model'
        self.meta_path = os.path.join(self.path, str(self.id) + '.meta')
        self.model_path = os.path.join(self.path, str(self.id) + '.pkl')

        # try to load metadata
        self._load_metadata()
        # try to load model
        self._load_model()

        self.target_column = list(self.target.keys())[0] if self.target else None

    def _load_metadata(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as meta_file:
                model_data = json.load(meta_file)
        else:
            model_data = {}

        self.target = model_data.get('target', None)
        self.features = model_data.get('features', {})
        self.labels = model_data.get('labels', {})
        self.encoding_index = model_data.get('encoding_index', 0)

    def _save_metadata(self):
        model_data = {'features': self.features,
                      'target': self.target,
                      'labels': self.labels,
                      'encoding_index': self.encoding_index}
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
        return self.pre_processing(df, create_label_encoding=True, remove_nan=True)

    def create_test_df(self, df):
        return self.pre_processing(df, create_label_encoding=False, remove_nan=True)

    def predict(self, df):
        if self.model is None:
            raise ValueError('No model is available for prediction')

        df = df.copy()

        df = self.pre_processing(df)
        if self.target_column in df.columns:
            del df[self.target_column]

        filter_mask = ~(df.isnull().any(axis=1))
        if filter_mask.any():
            predictions = self.model.predict(df[filter_mask])
        else:
            predictions = [np.nan] * len(df)

        df[self.target_column] = np.nan
        df.loc[filter_mask, self.target_column] = predictions
        self._post_processing(df)
        return df[self.target_column]

    def pre_processing(self, df, create_label_encoding=False, remove_nan=False):

        col_list = list(self.features.keys()) + list(self.target.keys())
        col_list = sorted(list(set([x for x in col_list if x in df.columns])))
        df = df[col_list].copy()

        for column, transforms in self.features.items():
            if not transforms:
                continue
            for transform in transforms:
                if transform == 'C':
                    df[column] = self._label_encoding(df[column], create_label_encoding=create_label_encoding)
                elif transform == 'log':
                    df[column] = np.log(df[column])
                else:
                    raise KeyError('Unknown transform option: ' + transform)
        if remove_nan:
            df = df[~(df.isnull().any(axis=1))]

        return df

    def _post_processing(self, df):
        transforms = list(self.target.values())[0]
        for transform in transforms:
            df[self.target_column] = self._back_transformation(transform, df[self.target_column])

    def _label_encoding(self, input_ser, create_label_encoding=False):
        if create_label_encoding:
            for x in input_ser.unique():
                if x != x:
                    continue
                x = _label_safe_value(x)
                if x not in self.labels:
                    self.labels[x] = self.encoding_index
                    self.encoding_index += 1
        result = [_label_check(y, self.labels) for y in input_ser]
        return np.array(result, dtype=np.int64)

    @staticmethod
    def _back_transformation(transform, input_ser):
        if transform == 'log':
            return np.exp(input_ser)

    def delete(self):
        rm_file_or_dir(self.meta_path)
        rm_file_or_dir(self.model_path)
