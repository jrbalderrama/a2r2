from pandas import DataFrame, Timedelta, Timestamp


# get next monday after a given number of weeks
def get_next_monday(
    dataframe: DataFrame,
    weeks: int,
) -> Timestamp:
    timedelta = Timedelta(7 * weeks - 1, unit="day")
    timestamp = dataframe.index.min() + timedelta
    return timestamp.normalize()
