import numpy as np
from sklearn import compose
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from ...learn import preprocessing


def make_column_transformer() -> ColumnTransformer:
    return compose.make_column_transformer(
        ## current buses dataset is not big enough to 'onehotencode' of 'month'
        (OneHotEncoder(handle_unknown="ignore", dtype=np.int_), ["dayofweek"]),
        (
            # OneHotEncoder(handle_unknown="ignore", drop="first", dtype=np.int_),
            OrdinalEncoder(dtype=np.int_),
            ["background"],
        ),
        (MinMaxScaler(), ["students"]),
        ## "workingday" has no impact (or slighlty neg) in score as 'ordinalencode'
        (OrdinalEncoder(dtype=np.int_), ["holiday"]),
        ## "index" has a significative negative impact in the model
        # (preprocessing.DatetimeIndexTransformer(15), ["index"]),
        ## 'spline transformer' outperform compared to 'trigonometric transformers'
        (preprocessing.periodic_spline_transformer(24, splines=12), ["hour"]),
        (preprocessing.periodic_spline_transformer(60, splines=30), ["minute"]),
        # (preprocessing.sin_transformer(24), ["hour"]),
        # (preprocessing.cos_transformer(24), ["hour"]),
        # (preprocessing.sin_transformer(60), ["minute"]),
        # (preprocessing.cos_transformer(60), ["minute"]),
        # remainder=MinMaxScaler(),
        remainder="drop",
        n_jobs=-1,
    )
