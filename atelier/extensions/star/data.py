from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Timestamp


# drop geospatial attributes from dataset
def tidy_dataframe(
    dataframe: DataFrame,
) -> DataFrame:
    return dataframe[
        [
            "departure_time",
            "id",
            "stop_name",
            "route_short_name",
            "stop_id",
            "direction_id",
        ]
    ].copy()


# query the dataset by attribute and value
def query(
    dataframe: DataFrame,
    name: str,
    value: Union[str, int, float, Sequence[str]],
) -> DataFrame:
    return (
        dataframe.query(f"{name} == {value}")
        if isinstance(value, (int, float))
        else dataframe.query(f'''{name} == "{value}"''')
        if isinstance(value, str)
        else dataframe.query(f"{name} in {value}")
    )


# filter dataset between two timestamps
# when 'start > end' then return the complement result (negation)
def between(
    dataframe: DataFrame,
    start: Union[str, Timestamp],
    end: Union[str, Timestamp],
    *,
    index: str = "departure_time",
) -> DataFrame:
    start_ = start if isinstance(start, Timestamp) else Timestamp(start)
    end_ = end if isinstance(end, Timestamp) else Timestamp(end)
    mask = (dataframe[index] >= start_) | (dataframe[index] < end_)
    if start_ > end_:
        mask = np.invert(mask)

    # dataframe.set_index(index).loc[start_:end_].reset_index()
    return dataframe.loc[mask]


# intersect two datasets with a common attribute ('on')
def intersect(
    right: DataFrame,
    left: DataFrame,
    on: Optional[Sequence[str]] = None,
    *,
    how: str = "inner",
) -> DataFrame:
    on_ = right.columns.values.tolist() if not on else on
    return pd.merge(right, left, how=how, on=on_)


# get distinct rows from a dataset grouping by a 'subset'
def distinct(
    dataframe: DataFrame,
    subset: Union[str, Sequence[str]],
) -> DataFrame:
    return dataframe.drop_duplicates(subset=subset)


# count rows by attribute name and value
def count(
    dataframe: DataFrame,
    attribute: str,
    value: Union[str, int, float],
    *,
    frequency: str = "15T",
    index: str = "departure_time",
) -> DataFrame:
    dataframe_ = (
        dataframe[dataframe[attribute] == value]
        .set_index(index)
        .groupby(
            [
                pd.Grouper(level=index, freq=frequency),
            ]
        )
        .count()
    )

    # #domain = pd.date_range(start=dataframe_.index.min(), end=dataframe_.index.max(), freq="15T")
    # #dataframe_ = dataframe_.reindex(domain, method=None, fill_value=NA)
    # #dataframe_.replace(0, np.NAN, inplace=True)
    # #display_dataframe(dataframe_)
    return dataframe_.iloc[:, 0].to_frame(name="count")
