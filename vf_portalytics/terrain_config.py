#!/usr/bin/env python

import numpy as np
import os
import subprocess

# ----------- paths & file format ----------
data_file_format = "parquet"
model_artifacts_path = os.path.expanduser(
    "~/onedrive/RGM/VF_integration/terrain-artifacts"
)
model_data_path = os.path.expanduser(
    "~/onedrive/RGM/VF_integration/terrain-integration-tables"
)
coefs_path = f"{model_data_path}/coefficients_cpe_by_prod_hash_with_mults.parquet"
account_id_mapper_path = f"{model_data_path}/account_id_to_cpe_map.parquet"
pid_mapper_path = f"{model_data_path}/pid_to_prod_hash_map.parquet"


# -------------- column names --------------
# VF call columns
account_id_col = "account_id"
baseline_col = "baseline_units_ext"
item_col = "pid"
start_date_col = "in_store_start_date"
end_date_col = "in_store_end_date"
discount_col = "promoted_price_ratio"
display_col = "on_display"
feature_col = "on_feature"
promo_features = [discount_col, display_col, feature_col]

# Columns in lookup tables
customer_col = "cpe_1"
prod_line_col = "prod_line_hash"
promo_coefs = ["discount_coef", "display_coef", "feature_coef"]

# ----------------- misc -----------------
try:
    _version = (
        subprocess.check_output(["git", "rev-parse", "--short=10", "HEAD"])
        .decode()
        .strip()
    )
    _version = f"_{_version[:8]}"
except subprocess.CalledProcessError:
    _version = ""

vf_id = f"terrain_module{_version}"


# ------------- model xforms -----------
def _xform_tpr(x, y):
    return x**y  # This is if they give x = consumer promo price / base price ratio
    # return (1 - x / 100)) ** y # This is if they give x as a price discount in %tage


def _xform_other(x, y):
    return np.exp(x * y)  # This is if they give feat/disp as x = proportion of stores
    # return np.exp((x / 100) * y) # This is if they give feat/disp as x = %tage of stores


transforms = {
    discount_col: _xform_tpr,
    display_col: _xform_other,
    feature_col: _xform_other,
}

# -------------- metadata --------------
_ordered_column_list = [
    account_id_col,
    item_col,
    start_date_col,
    end_date_col,
    baseline_col,
    *promo_features,
]
metadata = {
    "features": dict(zip(_ordered_column_list, [[]] * len(_ordered_column_list))),
    "target": "total_ef_qty",
    "labels": {},
    "one_hot_encode": False,
    "ordered_column_list": _ordered_column_list,
}
