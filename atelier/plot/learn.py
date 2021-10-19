from typing import Tuple

import numpy as np
import plotly.express as px
from pandas import DataFrame


def residuals_plot(
    dataframe: DataFrame,
    attributes: Tuple[str, str, str],
    *,
    size: int = 500,
) -> None:
    dataframe_ = dataframe.apply(abs).astype(np.int_)
    figure = px.scatter(
        dataframe,
        x=attributes[0],
        y=attributes[1],
        opacity=0.65,
        # size=attributes[2],
        # color=attributes[2],
        title="Residuals",
        marginal_y="box",
        # trendline="ols",
    )

    figure.update_coloraxes(showscale=False)
    # figure.update_scenes(aspectmode="data")
    # figure.update_yaxes(scaleanchor="x", scaleratio=1)
    figure.update_layout(width=size, height=size)
    figure.show()


def losses_plot(
    sequence,
    *,
    size: int = 500,
):
    figure = px.line(
        x=range(1, len(sequence) + 1),
        y=sequence,
        title="Losses",
        labels=dict(x="Epoch", y="Loss"),
        log_y=True,
    )

    # figure.update_yaxes(scaleanchor="x", scaleratio=1)
    figure.update_layout(width=size, height=size)
    figure.show()
