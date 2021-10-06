import importlib

import plotly.io as pio
from IPython import get_ipython

GOOGLE_COLAB = "google.colab"


def setup() -> bool:
    instance = get_ipython()
    registered = True if GOOGLE_COLAB in str(instance) else False
    if registered:
        pio.renderers.default = "colab"
        spec = importlib.util.find_spec(GOOGLE_COLAB)
        if spec:
            module = ".".join([GOOGLE_COLAB, "data_table"])
            data_table = importlib.import_module(module)
            enable_dataframe_formatter = getattr(
                data_table,
                "enable_dataframe_formatter",
            )

            enable_dataframe_formatter()

    return registered
