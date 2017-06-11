import numpy as np
import os
import shutil


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


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def forecast_error(real_lift, predicted_lift, baseline_units):
    predicted_units = baseline_units * predicted_lift
    real_units = baseline_units * real_lift
    error_units = abs(real_units - predicted_units)
    return (sum(error_units) / sum(real_units)) * 100
