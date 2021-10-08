from typing import Union

import numpy as np
import pandas as pd
from pandas import NA, DataFrame, Timedelta, Timestamp


# bucketize attribute
def onehot_encode(
    dataframe: DataFrame,
    column: str,
) -> DataFrame:
    dummies = pd.get_dummies(
        dataframe[column],
        prefix=column,
    )

    return pd.concat(
        [dataframe, dummies],
        axis=1,
    ).drop(columns=[column])


# encode (time) column as periodic wave
def periodic_encode(
    dataframe: DataFrame,
    column: str,
    period: int,
    start_num: int = 0,
) -> DataFrame:
    kwargs = {
        f"sin_{column}": lambda x: np.sin(
            2 * np.pi * (dataframe[column] - start_num) / period
        ),
        f"cos_{column}": lambda x: np.cos(
            2 * np.pi * (dataframe[column] - start_num) / period
        ),
    }

    return dataframe.assign(**kwargs).drop(columns=[column])


# mark dataset  between range(start, delta) with label+value
def label_between(
    dataframe: DataFrame,
    timestamp: Timestamp,
    timedelta: Timedelta,
    *,
    label: str,
    value: Union[int, str, float, bool],
) -> DataFrame:
    dataframe_ = dataframe.copy()
    dataframe_[label] = NA
    timestamp_ = timestamp + timedelta
    labels = (dataframe_.index >= timestamp) & (dataframe_.index < timestamp_)
    dataframe_.loc[labels, label] = value
    return dataframe_
