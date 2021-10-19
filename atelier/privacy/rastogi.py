import math
from typing import Optional

import numpy as np
from numpy import linalg, ndarray
from pandas import NA, DataFrame, Series


# Perturb timeline series with differential privacy
def fpa(Q: ndarray, δ: float, ε: float, k: int) -> ndarray:

    # define a laplace mechanism for perturbation
    def lpa(Q: ndarray, δ: float, ε: float) -> ndarray:
        # differential privacy scale based on the budget
        λ = δ / ε

        # Laplace mechanism applied to whole serie
        Z = np.random.laplace(scale=λ, size=Q.size)

        return Q + Z

    # discrete Fourier trasform
    F = np.fft.fft(Q)

    # first k values of DFT
    F_k = F[:k]

    # lpa of F_k
    Fλ_k = lpa(F_k.real, δ, ε) + 1j * lpa(F_k.imag, δ, ε)

    # Fλ_k with `n - k` zero-padding
    Fλ_n = np.pad(Fλ_k, (0, Q.size - k))

    # inverse discrete Fourier transform
    Qλ = np.fft.ifft(Fλ_n)

    # modulus of complex values of IFFT
    Qλ_m = np.absolute(Qλ)

    # round perturbation to integers
    Qλ_int = np.rint(Qλ_m)

    # replace negative values with zeroes
    Qλ_int[Qλ_int < 0] = 0

    return Qλ_int


# perform a noise perturbation with the Rastogi algorithm
def fourier_perturbation(
    sequence: Series,
    boundary: float,
    budget: float,
    coefficients: int,
) -> Optional[ndarray]:

    # calculate the L-norm of a uniform vector of seed values
    def norm(seed: float, size: int, order: int) -> float:
        serie = np.full((size,), seed)
        return linalg.norm(serie, order)

    size = sequence.size
    if size > coefficients:
        sensitivity = math.sqrt(coefficients) * norm(boundary, size, 2)
        return fpa(
            sequence.to_numpy(),
            sensitivity,
            budget,
            coefficients,
        )

    return None


def fourier_perturbation_by_timeframe(
    sequence: Series,
    boundary: float,
    epsilon: float,
    coefficients: int,
    *,
    period: str = "week",
) -> ndarray:
    def get_period(dataframe: DataFrame, period: str) -> Optional[Series]:
        return {
            "day": dataframe.index.date,
            "week": dataframe.index.isocalendar().week,
            "month": dataframe.index.month_name(),
            "year": dataframe.index.year,
        }.get(period)

    dataframe = sequence.copy().to_frame()
    dataframe["period"] = get_period(dataframe, period)
    fpas = []
    for period in dataframe["period"].unique():
        period_sequence = dataframe[dataframe["period"] == period]
        period_fpa = fourier_perturbation(
            period_sequence.iloc[:, 0],
            boundary,
            epsilon,
            coefficients,
        )

        fpas.append(period_fpa)

    dataframe["fpa"] = np.concatenate(fpas).ravel()
    return dataframe["fpa"].to_numpy()


def bound(
    serie: Series,
    aggregate: str,
) -> float:
    def ceil(serie: Series) -> float:
        maximum = serie.max()
        # maximum = linalg.norm(Q, np.inf)
        # # round(maximum, -1)
        return 10 * math.ceil(maximum / 10)

    return {
        "count": 1,
        "sum": ceil(serie),
    }.get(aggregate, NA)