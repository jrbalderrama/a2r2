import numpy as np
from pandas import DataFrame, Series


# compute the entropy of a serie
def entropy(
    series: Series,
    base: int = 2,
    normalize: bool = False,
) -> float:

    # compute the expectation of a serie
    def expectation(probability: Series) -> float:
        return (probability * np.log(probability) / np.log(base)).sum()

    # compute the efficiency of a serie
    def efficiency(entropy: float, length: int) -> float:
        return entropy * np.log(base) / np.log(length)

    probability = series.value_counts(normalize=True, sort=False)
    h = -expectation(probability)
    _entropy = efficiency(h, series.size) if normalize else h
    return _entropy


# compute the entropy of a dataframe
def get_entropies(
    dataframe: DataFrame,
    *,
    base: int = 2,
    normalize: bool = False,
) -> Series:
    dataframe_ = dataframe.copy()
    entropies = dataframe_.apply(entropy, base=base, normalize=normalize)
    return (
        entropies.to_frame()
        .reset_index()
        .rename(
            {
                "index": "attribute",
                0: "entropy",
            },
            axis=1,
        )
    )
