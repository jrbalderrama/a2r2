from typing import Optional, Sequence, Tuple

import pandas as pd
from pandas import DataFrame, Timedelta, Timestamp
from plotly import subplots
from plotly.graph_objs import Bar, Candlestick, Figure, Scatter


# show a timeseries graph of a selected attribute
def plot(
    dataframe: DataFrame,
    attribute: str,
    *,
    title: Optional[str] = None,
) -> None:
    figure = Figure()
    scatter = Scatter(
        x=dataframe.index,
        y=dataframe[attribute],
        mode="lines",
        name="values",
        connectgaps=False,
    )

    figure.add_trace(scatter)
    figure.update_layout(
        showlegend=False,
        title_text=title or attribute,
        template="simple_white",
    )

    figure.update_xaxes(showgrid=True)
    figure.show()


# show timeline divided bt delimiters and holidays
def plot_with_annotations(
    dataframe: DataFrame,
    attributes: Sequence[str],
    *,
    delimiters: Optional[Sequence[Tuple[str, Timestamp]]] = None,
    timeframes: Optional[Sequence[Tuple[str, Timestamp, Timedelta]]] = None,
    title: Optional[str] = None,
    secondary_y: bool = True,
) -> None:
    specs = None
    if secondary_y:
        specs = [[{"secondary_y": secondary_y}]]

    figure = subplots.make_subplots(specs=specs)
    for counter, column in enumerate(attributes):
        right_y = False if counter % 2 == 0 else True
        scatter = Scatter(
            x=dataframe.index,
            y=dataframe[column],
            mode="lines",
            name=column,
        )

        if secondary_y:
            figure.add_trace(scatter, secondary_y=right_y)
        else:
            figure.add_trace(scatter)

    if delimiters:
        for name, timestamp in delimiters:
            figure.add_shape(
                x0=timestamp,
                x1=timestamp,
                xref="x",
                y0=0.9,
                yref="paper",
                y1=0,
                type="line",
                line=dict(
                    # color="Gray",
                    width=1,
                    dash="dashdot",
                ),
            )

            figure.add_annotation(
                x=timestamp,
                xref="x",
                y=0.9,
                yref="paper",
                text=name,
                showarrow=True,
                yshift=-15,
            )

    if timeframes:
        for name, timestamp, timedelta in timeframes:
            end_timeframe = timestamp + timedelta
            figure.add_shape(
                x0=timestamp,
                x1=end_timeframe,
                xref="x",
                y0=0,
                y1=1,
                yref="paper",
                type="rect",
                layer="below",
                fillcolor="LightSeaGreen",
            )

            figure.add_annotation(
                x=timestamp,
                xref="x",
                y=1,
                yref="paper",
                text=name,
                textangle=90,
                # align="right",
                showarrow=False,
                xshift=10,
                # yshift=-10,
            )

    # figure.update_shapes(dict(xref="x", yref="y"))
    figure.update_layout(
        title_text=title or "",
        dragmode="zoom",
        hovermode="x",
        template="simple_white",
        legend=dict(
            x=1,
            y=1,
            xanchor="right",
            yanchor="bottom",
            orientation="h",
        ),
    )

    figure.update_xaxes(
        range=[
            dataframe.index.min(),
            dataframe.index.max(),
        ],
        type="date",
    )

    figure.update_yaxes(rangemode="tozero")  # type="log",
    if secondary_y:
        figure.update_yaxes(title_text=attributes[0], secondary_y=False)
        figure.update_yaxes(title_text=attributes[1], secondary_y=True)

    figure.show()


# show residuals as kind of OHLC Charts
# attributes are tuple(truth, prediction) names of dataframe
def residuals_plot(
    dataframe: DataFrame,
    attributes: Tuple[str, str] = ("references", "predictions"),
    *,
    title: Optional[str] = None,
) -> None:
    hovertext = []
    index = dataframe.index
    references = dataframe[attributes[0]]
    predictions = dataframe[attributes[1]]
    for i in range(dataframe.shape[0]):
        hovertext.append(
            f"{index[i]}<br>"
            f"Reference: {references[i]}<br>"
            f"Prediction: {predictions[i]}"
        )

    figure = Figure(
        data=[
            Scatter(
                x=index,
                y=references,
                mode="lines",
                name="reference",
                line=dict(color="lightgrey", width=0.6, dash="dot"),
                showlegend=False,
            ),
            Scatter(
                x=index,
                y=predictions,
                mode="lines",
                name="prediction",
                line=dict(color="lightblue", width=0.6, dash="dot"),
                showlegend=False,
            ),
            Candlestick(
                x=index,
                open=references,
                high=predictions,
                low=predictions,
                close=references,
                text=hovertext,
                hoverinfo="text",
                name="residuals",
                # line=dict(width=2),
                increasing_line_color="green",
                decreasing_line_color="red",
                showlegend=False,
            ),
        ]
    )

    figure.update_layout(
        title=title or "Timeline Residuals",
        template="simple_white",
        xaxis_rangeslider_visible=False,
    )

    figure.show()


# zoom on predictions of a predictor and a baseline model with ground truth
def predictions_interval_plot(
    gt_dataframe: DataFrame,
    bl_dataframe: DataFrame,
    pm_dataframe: DataFrame,
    names: Tuple[str, str, str],
    *,
    title: Optional[str] = None,
) -> None:
    figure = Figure(
        data=[
            Scatter(
                x=gt_dataframe.index,
                y=gt_dataframe["validations"],
                mode="lines",
                name=names[0],
                # rgba(0,0,0, 0.3)
                line=dict(color="gray", width=1, dash="dot"),
            ),
            Scatter(
                x=bl_dataframe.index,
                y=bl_dataframe.predictions,
                mode="lines",
                name=names[1],
                opacity=0.8,
            ),
            Scatter(
                x=pm_dataframe.index,
                y=pm_dataframe.predictions,
                mode="lines",
                name=names[2],
                opacity=0.8,
                visible="legendonly",
            ),
        ]
    )

    figure.update_layout(
        showlegend=True,
        title_text=title or "Predictions",
        template="simple_white",
        xaxis=dict(
            range=[
                pm_dataframe.index.min(),
                pm_dataframe.index.max(),
            ],
        ),
    )

    figure.update_xaxes(rangeslider_visible=True)
    figure.show()


# TODO rework and check residuals diff between bar and scatters
def predictions_interval_plot_with_staggings(
    dataframe: DataFrame,
    staggered: DataFrame,
    names: Tuple[str, str, str],
    *,
    title: Optional[str] = None,
) -> None:
    figure = subplots.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
    )

    prediction_plot = Scatter(
        x=dataframe.index,
        y=dataframe.predictions,
        mode="lines",
        name=names[0],
        showlegend=False,
        fill=None,
        line=dict(color="gray", width=0.1),
    )

    figure.add_trace(prediction_plot, row=1, col=1)
    staggered_plot = Scatter(
        x=staggered.index,
        y=staggered.predictions,
        mode="lines",
        name=names[1],
        showlegend=False,
        fill="tonexty",
        fillcolor="blue",
        line=dict(color="gray", width=0.1),
        # line_color="gray",
        opacity=0.8,
        # hoverinfo="x+y",
        # stackgroup='one'
    )

    figure.add_trace(staggered_plot, row=1, col=1)
    residuals = (
        pd.merge(
            dataframe,
            staggered,
            how="inner",
            left_index=True,
            right_index=True,
        )
        .rename(
            {
                "predictions_x": "prediction",
                "predictions_y": "staggered",
            },
            axis=1,
        )
        .drop(["references_x", "references_y"], axis=1)
        .dropna()
        .astype(int)
    )

    residuals["difference"] = residuals["prediction"] - residuals["staggered"]
    # display(residuals)

    colors = ["green" if c > 0 else "red" for c in residuals["difference"]]
    bar_plot = Bar(
        x=residuals.index,
        y=residuals.difference,
        name="difference",
        showlegend=False,
        marker_color=colors,
    )

    figure.add_trace(bar_plot, row=2, col=1)
    figure.update_layout(
        showlegend=True,
        title_text=title or "Predictions and Staggings",
        template="simple_white",
    )

    figure.update_xaxes(showticklabels=True, row=1, col=1)
    figure.update_xaxes(showticklabels=False, row=2, col=1, visible=False)
    figure.update_yaxes(
        title_text="difference",
        row=2,
        col=1,
        zeroline=True,
        zerolinecolor="gray",
    )

    figure.show()
