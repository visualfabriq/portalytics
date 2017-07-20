import os
import pandas as pd
from vf_portalytics.tool import rm_file_or_dir


class DataSet(object):
    def __init__(self, id, data_df=None, path=None):
        """
        Class constructor
        :param id: id of data set
        :param data_df: data-frame of data
        :param description: description of data set
        :param name: name of data set

        :return:
        """
        if id is None:
            raise TypeError('A dataset id should be given')

        self.id = id
        self.path = path or '/srv/datasets/'
        if not os.path.exists(self.path):
            raise KeyError('Path ' + self.path + ' does not exist')

        self.file_path = os.path.join(self.path, 'dataset_' + str(id) + '.msgpack')

        if data_df is None:
            # load data from disk
            self._load_data()
        else:
            # load data from data_df
            self.data_df = data_df
            self._save_data()

    def __str__(self):
        return self.id

    def _load_data(self):
        self.data_df = pd.read_msgpack(self.file_path)

    def _save_data(self):
        self.data_df.to_msgpack(self.file_path)

    def delete(self):
        rm_file_or_dir(self.file_path)

    def save(self):
        self._save_data()

