from typing import Callable, Optional, Sequence, Union

from pandas import DataFrame


# pre processing transportation data
def pre_process_by_aggregation(
    dataframe: DataFrame,
    *,
    stop_names: Optional[Sequence[str]],
    ignore_weekend: bool = False,
    agg: Union[str, Callable] = sum,
) -> DataFrame:

    dataframe_ = dataframe.copy()
    # filter data from 'bus_stops' only
    if stop_names:
        dataframe_ = dataframe_[dataframe_["stop_name"].isin(stop_names)]

    # remove weekend information
    if ignore_weekend:
        dataframe_ = dataframe_.set_index("departure_time")
        dataframe_ = dataframe_[dataframe_.index.dayofweek < 5]

    # first aggregate dataset by stop name and departure time
    # in order to preserve (departure_time' as distinct index
    dataframe_ = (
        dataframe_.groupby(
            [
                "stop_name",
                "departure_time",
            ]
        )
        .agg({"count": "sum"})
        .reset_index()
    )

    return dataframe_.groupby("departure_time").aggregate(agg)
