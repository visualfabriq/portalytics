import os
from pandas.api.types import is_numeric_dtype
import pandas as pd
import collections
from vf_portalytics.tool import rm_file_or_dir


def is_var_hashable(var_values):
    for val in var_values:
        if not isinstance(val, collections.Hashable):
            return False

    return True


def detect_vars_type(data):
    data_types_dict = data.dtypes.to_dict()
    numerical_var_list = []
    categorical_var_list = []
    type_dict = {}

    for key, value in data_types_dict.iteritems():
        if is_numeric_dtype(value):
            type_dict[key] = 'Numerical'
            numerical_var_list.append(key)
        elif value == object:
            is_hash = is_var_hashable(var_values=data[key].values)
            if is_hash:
                type_dict[key] = 'Categorical'
                categorical_var_list.append(key)
            else:
                type_dict[key] = 'Other'
                print 'Variable {} is not hashable'.format(key)
                continue
        else:
            type_dict[key] = 'Other'  # mostly datetime dtype
            # TODO: add 'Time Series' data type, maybe use: data.var_name.is_time_series

    return type_dict, categorical_var_list, numerical_var_list


def variable_type_list(variables):
    type_dict = dict()
    categorical_var_list = list()
    numerical_var_list = list()

    for variable in variables:
        type_dict[variable['name']] = variable['type']
        if variable['type'] == 'Numerical':
            numerical_var_list.append(variable['name'])
        elif variable['type'] == 'Categorical':
            categorical_var_list.append(variable['name'])

    return type_dict, categorical_var_list, numerical_var_list


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

        type_dict, categorical_var_list, numerical_var_list = detect_vars_type(self.data_df)

        self.type_dict = type_dict
        self.categorical_var_list = categorical_var_list
        self.numerical_var_list = numerical_var_list

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

