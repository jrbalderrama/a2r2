from typing import Tuple

import pandas as pd
from pandas import DataFrame, Timedelta


# Merge datasets with datetime index
def merge_dataframes(
    left: DataFrame,
    right: DataFrame,
    *,
    how: str = "outer",
) -> DataFrame:

    left_max = left.index.max()
    right_max = right.index.max()
    farest = right_max if right_max <= left_max else left_max
    farest += Timedelta(1, unit="day")
    left_ = left[left.index <= farest]
    right_ = right[right.index <= farest]

    # merge datasets
    dataframe = pd.merge(
        left_,
        right_,
        how=how,
        left_index=True,
        right_index=True,
    )

    # fill empty values with zeroes
    dataframe = dataframe.fillna(0)
    return dataframe


# split the dataframe in inputs and outputs (X, y) for `fit`
def split_dataframe(
    dataframe: DataFrame,
    target: str,
) -> Tuple[DataFrame, DataFrame]:
    y = dataframe[[target]]
    X = dataframe.drop(columns=[target])
    return X, y
