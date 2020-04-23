# coding=utf-8
import numpy as np
import os
import shutil
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd


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


def describe_columns(df, output=True):
    generic = []
    tag = []
    promotion_dimension = []
    field = []
    product = []
    product_dimension = []
    media = []
    media_attr = []

    for col in df.columns:
        if col.startswith('product_dimension'):
            product_dimension.append(col)
        elif col.startswith('product_'):
            product.append(col)
        elif col.startswith('media_'):
            media.append(col)
        elif col.startswith('mediaattr_'):
            media_attr.append(col)
        elif col.startswith('promotion_dimension_'):
            promotion_dimension.append(col)
        elif col.startswith('tag_'):
            tag.append(col)
        elif col.startswith('field_'):
            field.append(col)
        else:
            generic.append(col)

    if output:
        if generic:
            print('\nStandard Columns: ')
            print(', '.join(generic))
        if field:
            print('\nPromotion Fields: ')
            print(', '.join(field))
        if tag:
            print('\nTags: ')
            print(', '.join(tag))
        if promotion_dimension:
            print('\nPromotion Dimensions: ')
            print(', '.join(promotion_dimension))
        if product:
            print('\nProduct Attributes: ')
            print(', '.join(product))
        if product_dimension:
            print('\nProduct Dimensions: ')
            print(', '.join(product_dimension))
        if media:
            print('\nUsed Media: ')
            print(', '.join(media))
        if media_attr:
            print('\nMedia Attributes: ')
            print(', '.join(media_attr))

    return generic, field, tag, promotion_dimension, product_dimension, media, media_attr


def create_train_test_sets(df, mask, prediction_model, prediction_target='lift', debug=True):
    # convert
    train_df = df[mask]
    train_lift = train_df[prediction_target]
    del train_df[prediction_target]
    train_df = prediction_model.pre_processing(train_df, train_mode=True)

    test_df = df[~mask]
    test_lift = test_df[prediction_target]
    del test_df[prediction_target]
    test_df = prediction_model.pre_processing(test_df, train_mode=False)

    if debug:
        print('Train set: ' + str(len(train_df)))
        print('Test set: ' + str(len(test_df)))

    return train_df, train_lift, test_df, test_lift


def mean_absolute_percentage_error(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def forecast_error(predicted_lift, real_lift, baseline_units):
    predicted_units = baseline_units * predicted_lift
    real_units = baseline_units * real_lift
    error_units = abs(real_units - predicted_units)
    return (sum(error_units) / sum(real_units)) * 100


def score_model(predict_lift, test_lift, baseline=None):
    print("R²:" + str(r2_score(predict_lift, test_lift)))
    print("MAE:" + str(mean_absolute_error(predict_lift, test_lift)))
    print("MAPE:" + str(mean_absolute_percentage_error(predict_lift, test_lift)))
    if baseline is not None:
        print("Forecast error:" + str(forecast_error(predict_lift, test_lift, baseline)))


def squared_error_objective_with_weighting(y_pred, y_true, under_predict_weight=1.0, over_predict_weight=1.0):
    """
    Return Gradient and Hessian of Mean Squared error objective function with different weights for under and
    overshooting the target.
    """
    grad = y_true - y_pred

    grad = np.where(grad > 0, grad * under_predict_weight, grad * over_predict_weight)
    hess = np.full(y_true.shape, 1.0)
    return grad, hess


def get_categorical_features(data=None, potential_cat_feat=None):
    """
    Given a DataFrame
    Return a list of features that are potentially categorical
    If potential_cat_feat is given returns it back
    """
    if potential_cat_feat is None and isinstance(data, pd.DataFrame):
        potential_cat_feat = set(data.select_dtypes(include=['object']).columns)
        potential_cat_feat.update([feat_name for feat_name, row in data.items()
                                   if 2 <= len(row.unique()) < 30])
    return list(potential_cat_feat)
