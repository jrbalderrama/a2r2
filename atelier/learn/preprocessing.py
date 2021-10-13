from typing import Optional, Tuple, Union

import numpy as np
from pandas import DataFrame, DatetimeIndex, Timedelta, Timestamp
from pandas.tseries.offsets import BDay
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, SplineTransformer

from ..utils.time import HolidayCalendar


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def periodic_spline_transformer(period, splines=None, degree=3):
    if not splines:
        splines = period
    knots = splines + 1
    return SplineTransformer(
        degree=degree,
        n_knots=knots,
        knots=np.linspace(0, period, knots).reshape(knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )


class DatetimeIndexTransformer(TransformerMixin, BaseEstimator):
    """
    >>> dit = preprocessing.DatetimeIndexTransformer(15)
    >>> tin = dit.fit_transform(dataset.index)
    >>> inv = it.inverse_transform(tin)
    >>> idx = pd.Int64Index([1, 2, 3])
    >>> inv_idx = dit.inverse_transform(idx)
    """

    def __init__(self, frequence, unit="T", dtype=np.int_):
        self.dtype = dtype
        self.frequence = frequence
        self.minimum = None
        self.timedelta = Timedelta(frequence, unit=unit)
        self.unit = unit

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.minimum = X.min()
        X_ = (X - self.minimum) // self.timedelta
        return X_  # .view(self.dtype)

    def inverse_transform(self, X):
        return X * self.timedelta + self.minimum
        # return X_.view("datetime64[ns]")


def timeline_feature_extraction(
    dataframe: DataFrame,
    timeframes: Optional[Union[Timestamp, DatetimeIndex]] = None,
) -> DataFrame:
    datetime_index = dataframe.index
    index_series = datetime_index.to_series()
    working_days = BDay()
    dataframe_ = (
        dataframe.assign(hour=datetime_index.hour)
        .assign(minute=datetime_index.minute)
        .assign(dayofweek=datetime_index.dayofweek)
        .assign(day=datetime_index.day)
        .assign(month=datetime_index.month)
        .assign(workingday=index_series.apply(working_days.is_on_offset))
    )

    if timeframes is not None:
        calendar = HolidayCalendar(timeframes)
        holidays = calendar.holidays()
        dataframe_ = dataframe_.assign(
            holiday=index_series.dt.date.astype("datetime64").isin(holidays)
        )

    return dataframe_


# train+test split a dataframe
## if train_t is provided the data in gap between train_t and test_t
## is provided as second item of resulting tuple (useful for validation)
def timeline_train_test_split(
    dataframe: DataFrame,
    *,
    start_test: Timestamp,
    end_train: Optional[Timestamp] = None,
) -> Tuple[DataFrame, ...]:
    index = dataframe.index
    test = dataframe[index >= start_test]
    train = dataframe[index < start_test]
    mask = None
    if end_train:
        train = dataframe[index < end_train]
        mask = (index >= end_train) & (index < start_test)

    return (train, dataframe[mask], test) if (mask is not None) else (train, test)
