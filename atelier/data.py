from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from pandas import DataFrame, Timedelta


# Merge datasets with datetime index
def merge_dataframes(
    left: DataFrame,
    right: DataFrame,
    *,
    how: str = "outer",
    fillna: bool = True,
    fillna_value: Union[str, float, int, bool] = 0,
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
    if fillna:
        dataframe = dataframe.fillna(fillna_value)

    return dataframe


# split the dataframe in inputs and outputs (X, y) for `fit`
def split_dataframe(
    dataframe: DataFrame,
    target: str,
    reset_index: bool = False,
) -> Tuple[DataFrame, DataFrame]:
    y = dataframe[[target]]
    X = dataframe.drop(columns=[target])
    if reset_index:
        X = X.reset_index()

    return X, y


# aggregate dataframe wrapper
## if attribute is set select only rows with 'value' of the 'attribute' col
def aggregate_dataframe(
    dataframe: DataFrame,
    *,
    by: Union[str, List[str]],
    agg: Union[str, Callable, Dict[str, Union[str, Callable]]],
    attribute: Optional[str] = None,
    value: Optional[Union[str, List[str]]] = None,
    keep_index: bool = True,
) -> DataFrame:
    dataframe_ = dataframe.copy()
    if attribute and value:
        values = [value] if isinstance(value, str) else value
        dataframe_ = dataframe_[dataframe_[attribute].isin(values)]

    # TODO fix/check (?) reindex() method
    if attribute and keep_index:
        by_ = [by] if isinstance(by, str) else by
        by_.append(attribute)

        # first groupby to keep all times of 'attribute'
        dataframe_ = dataframe_.groupby(by_).agg(agg).reset_index()

    # # remove weekend information
    # dataframe_ = dataframe_[dataframe_.index.dayofweek < 5]
    return dataframe_.groupby(by).aggregate(agg)


# shift datetimeindex by delta
## if attribute is set shift only value associated to it
def shift_datetime_index(
    dataframe: DataFrame,
    timedelta: Timedelta,
    *,
    attribute: Optional[str] = None,
    value: Optional[Union[str, List[str]]] = None,
) -> DataFrame:
    dataframe_ = dataframe.copy()
    dataframe_.reset_index(inplace=True)
    index = dataframe_.columns[0]
    if attribute and value:
        values = [value] if isinstance(value, str) else value
        for value in values:
            labels = dataframe_[attribute] == value
            dataframe_.loc[labels, index] += timedelta
    else:
        dataframe_.loc[:, index] += timedelta

    dataframe_.set_index(index, inplace=True)
    return dataframe_
