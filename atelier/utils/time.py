from typing import Union

from pandas import DataFrame, DatetimeIndex, Timedelta, Timestamp
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday


# get next monday after a given number of weeks
def get_next_monday(
    dataframe: DataFrame,
    weeks: int,
) -> Timestamp:
    timedelta = Timedelta(7 * weeks - 1, unit="day")
    timestamp = dataframe.index.min() + timedelta
    return timestamp.normalize()


class HolidayCalendar(AbstractHolidayCalendar):
    def __init__(
        self,
        timeframes: Union[Timestamp, DatetimeIndex],
    ):
        super().__init__()
        for holiday in timeframes:
            if isinstance(holiday, Timestamp):
                self.rules.append(
                    Holiday(
                        "",
                        holiday.year,
                        holiday.month,
                        holiday.day,
                    )
                )
            else:
                holidays = [Holiday("", t.year, t.month, t.day) for t in holiday]
                self.rules.append(holidays)
