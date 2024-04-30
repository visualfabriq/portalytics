import numpy as np
import os


# -------------- column names --------------
# VF call columns
ACCOUNT_ID_COL = "account_id"
BASELINE_COL = "baseline_units_ext"
ITEM_COL = "pid"
START_DATE_COL = "in_store_start_date"
END_DATE_COL = "in_store_end_date"
DISCOUNT_COL = "promoted_price_ratio"
DISPLAY_COL = "on_display"
FEATURE_COL = "on_feature"
PROMO_FEATURES = [DISCOUNT_COL, DISPLAY_COL, FEATURE_COL]

# Columns in lookup tables
CUSTOMER_COL = "cpe_1"
PROD_LINE_COL = "prod_line_hash"
PROMO_COEFS = ["discount_coef", "display_coef", "feature_coef"]


# ------------- model xforms -----------
def _xform_tpr(x, y):
    return x**y  # This is if they give x = consumer promo price / base price ratio


def _xform_other(x, y):
    return np.exp(x * y)  # This is if they give feat/disp as x = proportion of stores


transforms = {
    DISCOUNT_COL: _xform_tpr,
    DISPLAY_COL: _xform_other,
    FEATURE_COL: _xform_other,
}

# -------------- metadata --------------
_ordered_column_list = [
    ACCOUNT_ID_COL,
    ITEM_COL,
    START_DATE_COL,
    END_DATE_COL,
    BASELINE_COL,
]
_ordered_column_list.extend(PROMO_FEATURES)
metadata = {
    "features": dict(zip(_ordered_column_list, [[]] * len(_ordered_column_list))),
    "target": {"total_ef_qty": []},
    "labels": {},
    "one_hot_encode": False,
    "ordered_column_list": _ordered_column_list,
}
