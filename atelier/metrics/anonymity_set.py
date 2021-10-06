import copy
from typing import List, Optional

from pandas import DataFrame, Series


def anoymity_set(
    dataframe: DataFrame,
    subset: Optional[List[str]] = None,
    reindex: bool = False,
) -> DataFrame:

    # reset index by including zeroes values
    def reset_index(serie: Series) -> Series:
        domain = range(1, serie.index.max() + 1)
        return serie.reindex(domain, fill_value=0)

    multiplicity = dataframe.value_counts(subset=subset)
    _anonimity_set = multiplicity.value_counts().sort_index()
    if reindex:
        _anonimity_set = reset_index(_anonimity_set)

    return _anonimity_set


# compute the anonymity set of a 'formated' dataframe
def get_anonymity_sets(
    dataframe: DataFrame,
    distinct: Optional[str] = None,
    *,
    subset: Optional[List[str]] = None,
    reindex: bool = False,
) -> Series:

    # select distinct columns by a defined attribute
    def get_distinct(
        dataframe: DataFrame,
        subset: Optional[List[str]] = None,
        distinct: Optional[str] = None,
    ) -> DataFrame:
        dataframe_ = dataframe.copy()
        if distinct:
            subset_ = copy.deepcopy(subset)
            if subset_:
                if distinct not in subset_:
                    subset_.append(distinct)
            else:
                subset_ = [distinct]

            dataframe_.drop_duplicates(subset=subset_, inplace=True)

        return dataframe_

    subset_ = None if not subset else subset
    dataframe_ = get_distinct(dataframe, subset_, distinct)
    anonymity_sets = anoymity_set(dataframe_, subset_, reindex)
    return (
        anonymity_sets.to_frame()
        .reset_index()
        .rename(
            {
                "index": "cardinality",
                0: "occurrences",
            },
            axis=1,
        )
    )
