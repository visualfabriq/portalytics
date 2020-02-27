import pandas as pd

import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from vf_portalytics.model import PredictionModel


def test_prediction_model(tmpdir):
    # Create a model wrapper
    model_dir = str(tmpdir.mkdir('model_data'))
    model_wrapper = PredictionModel('test_model', path=model_dir)

    # Generate test data
    x, y = make_regression(n_samples=1000, n_features=5, n_informative=3, random_state=1234)
    feature_names = ['feature_{}'.format(n) for n in range(1, 6)]
    x_total = pd.DataFrame(data=x, columns=feature_names)
    y_total = pd.Series(y, name='y_true')

    # Make train-test split and generate predictions
    train_idx, test_idx = train_test_split(x_total.index, random_state=5)
    x_train, y_train = x_total.loc[train_idx, :], y_total.loc[train_idx]
    x_test, y_test = x_total.loc[test_idx, :], y_total.loc[test_idx]

    linear_regression = LinearRegression().fit(x_train, y_train)
    y_pred = pd.Series(
        data=linear_regression.predict(x_test),
        name='y_pred',
        index=y_test.index
    )

    # Calculate scores
    score = abs(r2_score(y_test.values, y_pred.values))
    assert score > 0.8

    # Save the trained model
    model_wrapper.features = {c: [] for c in x_total.columns}
    model_wrapper.model = linear_regression
    model_wrapper.target = {'target': []}
    model_wrapper.ordered_column_list = sorted(model_wrapper.features.keys())

    # Save the model
    model_wrapper.save()

    # Reload the saved model
    saved_model = PredictionModel('test_model', path=model_dir)
    assert saved_model.features == model_wrapper.features
    assert saved_model.ordered_column_list == model_wrapper.ordered_column_list
    assert saved_model.target == model_wrapper.target

    # Make predictions with the saved model
    saved_predictions = pd.Series(
        data=saved_model.predict(x_test),
        name='y_pred',
        index=y_test.index
    )

    # Compare with the score we got before
    saved_model_score = abs(r2_score(saved_predictions.values, y_test.values))
    assert saved_model_score == pytest.approx(score)