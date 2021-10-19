import functools
import itertools
from typing import Optional, Sequence, Union

from pandas import DataFrame

from ...privacy import rastogi


def get_fourier_perturbations(
    dataframe: DataFrame,
    agg_sizes: Sequence[int],
    coefficients: Sequence[int],
    epsilons: Sequence[float],
    *,
    agg: str = "count",
    attribute: Optional[str] = None,
    value: Optional[Union[str, Sequence[str]]] = None,
    period: Optional[str] = None,
) -> DataFrame:
    dataframe_ = dataframe.copy()
    if attribute and value:
        values = [value] if isinstance(value, str) else value
        dataframe_ = dataframe_[dataframe_[attribute].isin(values)]

    # count validations (per user and timestamp)
    dataframe_ = (
        dataframe_.groupby(["id", "departure_time"])
        .agg({"count": agg})
        .rename(columns={"count": "validations"})
        .reset_index()
    )

    samples = DataFrame()
    for n in agg_sizes:
        subset = dataframe_["id"].drop_duplicates().sample(n).values
        mask = dataframe_["id"].isin(subset)
        sample = dataframe_[mask].reset_index(drop=True)
        sample = sample.assign(n=n).drop("id", axis=1)
        samples = samples.append(sample)

    fpas = DataFrame()
    perturbation_function = functools.partial(
        rastogi.fourier_perturbation_by_timeframe,
        period=period,
    )

    perturbation_function = (
        rastogi.fourier_perturbation
        if not period
        else functools.partial(
            rastogi.fourier_perturbation_by_timeframe,
            period=period,
        )
    )

    for n in agg_sizes:
        sample = samples.query(f"n=={n}")
        reference = sample.groupby("departure_time").agg(agg)
        boundary = rastogi.bound(sample["validations"], agg)
        for k, ε in itertools.product(coefficients, epsilons):
            iteration = reference.copy()
            iteration = iteration.assign(n=n, ε=ε, k=k)
            iteration["fpa"] = perturbation_function(
                iteration["validations"],
                boundary,
                ε,
                k,
            )

            iteration["noise"] = iteration["fpa"] - iteration["validations"]
            fpas = fpas.append(iteration)

    return fpas
