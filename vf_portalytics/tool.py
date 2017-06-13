# coding=utf-8
import numpy as np
import os
import shutil
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import pylab as plt
sns.set()


def rm_file_or_dir(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            if os.path.islink(path):
                os.unlink(path)
            else:
                shutil.rmtree(path)
        else:
            if os.path.islink(path):
                os.unlink(path)
            else:
                os.remove(path)


def create_train_test_sets(df, mask, prediction_model, prediction_target='lift', debug=True):
    # convert
    train_df = prediction_model.pre_processing(df[mask], create_label_encoding=True, remove_nan=True)
    train_lift = train_df[prediction_target]
    del train_df[prediction_target]

    test_df = prediction_model.pre_processing(df[-mask], create_label_encoding=True, remove_nan=True)
    test_lift = test_df[prediction_target]
    del test_df[prediction_target]

    if debug:
        print('Train set: ' + str(len(train_df)))
        print('Test set: ' + str(len(test_df)))

    return train_df, train_lift, test_df, test_lift


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def forecast_error(real_lift, predicted_lift, baseline_units):
    predicted_units = baseline_units * predicted_lift
    real_units = baseline_units * real_lift
    error_units = abs(real_units - predicted_units)
    return (sum(error_units) / sum(real_units)) * 100


def score_model(predict_lift, test_lift, baseline=None):
    print(u"R²:" + str(r2_score(predict_lift, test_lift)))
    print("MAE:" + str(mean_absolute_error(predict_lift, test_lift)))
    print("MAPE:" + str(mean_absolute_percentage_error(predict_lift, test_lift)))
    if baseline is not None:
        print("Forecast error:" + str(forecast_error(predict_lift, test_lift, baseline)))


def plot_feature_importance(regressor, train_df):
    sns.set(style="ticks")

    # Plot feature importance
    feature_importance = regressor.feature_importances_

    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    mask = feature_importance > 0.5

    feature_importance = feature_importance[mask]
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.figure(figsize=(18, 14), dpi=1000)
    plt.barh(pos, feature_importance[sorted_idx], align='center', color='#3FB265')

    plt.yticks(pos, train_df.columns[mask][sorted_idx], size=20)
    plt.xticks(size=20)
    plt.xlabel('Relative Importance', fontsize=20)

    sns.despine()
    plt.show()