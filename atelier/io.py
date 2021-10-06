import os
from errno import ENOENT
from pathlib import Path

import pyarrow.parquet as pq
from pandas import DataFrame


# load dataset from file system
def read_data(
    path: Path,
) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            ENOENT,
            os.strerror(ENOENT),
            path,
        )

    table = pq.read_table(path)
    return table.to_pandas()
