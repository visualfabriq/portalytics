import pandas as pd

import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from vf_portalytics.model import PredictionModel


def test_prediction_model_categorical_features(tmpdir):
    # Needs to be fixed
    # Generate test data
    x, y = make_regression(n_samples=1000, n_features=5, n_informative=3, random_state=1234)
    feature_names = ['feature_{}'.format(n) for n in range(1, 6)]
    x_total = pd.DataFrame(data=x, columns=feature_names)
    y_total = pd.Series(y, name='y_true')
    # Convert continuous variable to categorical
    x_total['feature_1'] = pd.cut(x_total['feature_1'], 3, labels=['A', 'B', 'C'])

    # Make train-test split and generate predictions
    train_idx, test_idx = train_test_split(x_total.index, random_state=5)
    x_train, y_train = x_total.loc[train_idx, :], y_total.loc[train_idx]
    x_test, y_test = x_total.loc[test_idx, :], y_total.loc[test_idx]

    # Use portalytics for preprocessing and training
    model_dir = str(tmpdir.mkdir('model_data'))
    model_wrapper = PredictionModel('test_model', path=model_dir, one_hot_encode=False)
    model_wrapper.features = {c: [] for c in x_total.columns}
    model_wrapper.features['feature_1'] = 'C'
    model_wrapper.target = {'target': []}

    x_train_portalytics = model_wrapper.pre_processing(x_train, train_mode=True)
    x_test_portalytics = model_wrapper.pre_processing(x_test)

    linear_regression = LinearRegression().fit(x_train_portalytics, y_train)
    y_pred_portalytics = pd.Series(
        data=linear_regression.predict(x_test_portalytics),
        name='y_pred',
        index=y_test.index
    )
    # Use sklearn that keeps the order of the categorical feature as it is (A< B < C)
    le = preprocessing.LabelEncoder()
    x_train['feature_1'] = le.fit_transform(x_train['feature_1'])
    x_test['feature_1'] = le.fit_transform(x_test['feature_1'])

    linear_regression = LinearRegression().fit(x_train, y_train)
    y_pred = pd.Series(
        data=linear_regression.predict(x_test),
        name='y_pred',
        index=y_test.index
    )

    # Calculate scores
    score = abs(r2_score(y_pred_portalytics.values, y_pred.values))

def test_prediction_model_one_hot_encoding(tmpdir):
    # Test preprocessing of one hot encoding

    # Generate test data
    x, y = make_regression(n_samples=1000, n_features=5, n_informative=3, random_state=1234)
    feature_names = ['feature_{}'.format(n) for n in range(1, 6)]
    x_total = pd.DataFrame(data=x, columns=feature_names)
    y_total = pd.Series(y, name='y_true')
    # Convert continuous variable to categorical
    x_total['feature_1'] = pd.cut(x_total['feature_1'], 3, labels=['A', 'B', 'C'])

    # Make train-test split and generate predictions
    train_idx, test_idx = train_test_split(x_total.index, random_state=5)
    x_train, y_train = x_total.loc[train_idx, :], y_total.loc[train_idx]
    x_test, y_test = x_total.loc[test_idx, :], y_total.loc[test_idx]

    # Use portalytics for preprocessing and training
    model_dir = str(tmpdir.mkdir('model_data'))
    model_wrapper = PredictionModel('test_model', path=model_dir, one_hot_encode=True)
    model_wrapper.features = {c: [] for c in x_total.columns}
    model_wrapper.features['feature_1'] = 'C'
    model_wrapper.target = {'target': []}

    x_train_portalytics = model_wrapper.pre_processing(x_train, train_mode=True)
    x_test_portalytics = model_wrapper.pre_processing(x_test)

    linear_regression = LinearRegression().fit(x_train_portalytics, y_train)
    y_pred_portalytics = pd.Series(
        data=linear_regression.predict(x_test_portalytics),
        name='y_pred',
        index=y_test.index
    )

    # Use sklearn for preprocessing and training
    lb = preprocessing.LabelBinarizer()
    x_train = x_train.join(pd.DataFrame(lb.fit_transform(x_train['feature_1']),
                              columns=lb.classes_,
                              index=x_train.index))
    x_test = x_test.join(pd.DataFrame(lb.transform(x_test['feature_1']),
                              columns=lb.classes_,
                              index=x_test.index))

    del x_train['feature_1'], x_test['feature_1']

    linear_regression = LinearRegression().fit(x_train, y_train)
    y_pred = pd.Series(
        data=linear_regression.predict(x_test),
        name='y_pred',
        index=y_test.index
    )
    # Check if the predictions are similar
    score = abs(r2_score(y_pred_portalytics.values, y_pred.values))
    assert score > 0.99