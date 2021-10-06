from typing import Callable, Optional, Sequence, Union

from pandas import DataFrame, Series, Timedelta


# aggregate dataset
def pre_process_by_aggregation(
    dataframe: DataFrame,
    *,
    by: Union[str, Sequence[str]] = "fin_cours",
    agg: Union[str, Callable] = sum,
) -> DataFrame:
    return dataframe.groupby(by).aggregate(agg)


# shift (delta) time by background in dataset if any
def shift_time(
    dataframe: DataFrame,
    minutes: int,
    *,
    backgrounds: Optional[Union[str, Sequence[str]]],
    attribute: str = "filiere",
    index: str = "fin_cours",
) -> DataFrame:
    dataframe_ = dataframe.copy()
    timedelta = Timedelta(minutes, unit="T")
    dataframe_.reset_index(inplace=True)
    if backgrounds:
        backgrounds_ = [backgrounds] if isinstance(backgrounds, str) else backgrounds
        for background in backgrounds_:
            labels = dataframe_[attribute] == background
            dataframe_.loc[labels, index] += timedelta
    else:
        dataframe_.loc[:, index] += timedelta

    dataframe_.set_index(index, inplace=True)
    return dataframe_
