import os
import pandas as pd
from pandas.core.common import is_numeric_dtype
from bcolz import ctable
import collections
import tarfile
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
        self.file_path = os.path.join(self.path, 'dataset_' + str(id) + '.bcolz')
        self.package_path = os.path.join(self.path, 'dataset_' + str(id) + '.pck')

        if data_df is None:
            # load data from disk
            self._load_data()
        else:
            # load data from data_df
            self.data_df = data_df
            self._save_data()

        type_dict, categorical_var_list, numerical_var_list = detect_vars_type(data_df)

        self.type_dict = type_dict
        self.categorical_var_list = categorical_var_list
        self.numerical_var_list = numerical_var_list

    def __str__(self):
        return self.id

    def _load_data(self):
        # check if we do not have the file yet, but we do have a package
        if not os.path.exists(self.file_path) and os.path.exists(self.package_path):
            self.unpackage()

        # now load the table
        try:
            self.ct = ctable(rootdir=self.file_path, mode='r')
        except:
            raise KeyError('No correct bcolz file was found at ' + self.file_path)

        self.data_df = self.ct.to_dataframe()

    def _save_data(self):
        self.ct = ctable.fromdataframe(self.data_df, rootdir=self.file_path, mode='w')
        self.ct.flush()

    def delete(self):
        rm_file_or_dir(self.file_path)
        rm_file_or_dir(self.package_path)

    def save(self):
        self.save_data()

    def package(self):
        with tarfile.open(self.package_path, 'w') as tar:
            tar.add(self.file_path, arcname=self.path)

    def unpackage(self):
        with tarfile.open(self.package_path) as tar:
            tar.extractall(self.file_path)
        rm_file_or_dir(self.package_path)
