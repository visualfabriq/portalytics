from sklearn.externals import joblib
import numpy as np
import os.path
import os
from vf_portalytics.tool import rm_file_or_dir
import json


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

        self.target_column = self.target.keys()[0] if self.target else None
        self.create_label_encoding = None

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
            # if compressed:
            #     # unzip
            #     joblib.dump(self.model, self.model_path(), compress=0)
        else:
            self.model = None

    def _save_model(self, compress_level=3):
        if self.model is not None:
            joblib.dump(self.model, self.model_path, compress=compress_level)

    def save(self, compress_level=3):
        self._save_metadata()
        self._save_model(compress_level=compress_level)

    def create_train_df(self, df):
        return self._pre_processing(df, create_label_encoding=True, remove_nan=True)

    def create_test_df(self, df):
        return self._pre_processing(df, create_label_encoding=False, remove_nan=True)

    def predict(self, df):
        df = df.copy()

        self.pre_processing(df)
        if self.target_column in df.columns:
            del df[self.target_column]

        filter_mask = (df[self.features.keys()].isnull().sum(axis=1) == 0)
        if filter_mask.any():
            predictions = self.model.predict(df[filter_mask])
        else:
            predictions = [np.nan] * len(df)

        df[self.target_column] = np.nan
        df.loc[filter_mask, self.target_column] = predictions
        self._post_processing(df)
        return df[self.target_column]

    def pre_processing(self, df, create_label_encoding=False, remove_nan=False):
        self.create_label_encoding = create_label_encoding
        df = df[sorted(list(set([x for x in self.features.keys() + self.target.keys()
                                 if x in df.columns])))]
        for column, transforms in self.features.iteritems():
            if not transforms:
                continue
            for transform in transforms:
                df[column] = self._transformation(transform, df[column])
        if remove_nan:
            df = df[df[self.features.keys()].isnull().sum(axis=1) == 0]

        return df

    def _post_processing(self, df):
        transforms = self.target.values()[0]
        for transform in transforms:
            df[self.target_column] = self._back_transformation(transform, df[self.target_column])

    def _label_encoding(self, column):
        if self.create_label_encoding:
            for x in column.unique():
                if unicode(x).replace('.', '') not in self.labels and x == x:
                    self.labels[unicode(x).replace('.', '')] = self.encoding_index
                    self.encoding_index += 1

        return column.apply(lambda y: self.labels[unicode(y).replace('.', '')]
        if y == y and unicode(y).replace('.', '') in self.labels else -1)

    def _transformation(self, transform, column):
        if transform == 'C':
            return self._label_encoding(column)
        elif transform == 'log':
            return column.apply(np.log)

    @staticmethod
    def _back_transformation(transform, column):
        if transform == 'log':
            return column.apply(np.exp)

    def package_model(self):
        pass

    def delete(self):
        rm_file_or_dir(self.meta_path)
        rm_file_or_dir(self.model_path)

    def enable_model(self):
        pass
        #     db.user_db.active_model_settings.remove()
        #     db.user_db.active_model_settings.insert({'features': self.features, 'target': self.target,
        #                                              'modelid': self.model_id, 'scikit_model': True})
