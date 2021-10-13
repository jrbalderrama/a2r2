from typing import Optional, Sequence, Tuple

import pandas as pd
from pandas import DataFrame, Timedelta, Timestamp
from plotly import subplots
from plotly.graph_objs import Bar, Candlestick, Figure, Scatter


# show a timeseries graph of a selected attribute
def timeline_plot(
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
def timeline_plot_with_annotations(
    dataframe: DataFrame,
    attributes: Sequence[str],
    *,
    delimiters: Optional[Sequence[Tuple[str, Timestamp]]] = None,
    timeframes: Optional[Sequence[Tuple[str, Timestamp, Timedelta]]] = None,
    title: Optional[str] = None,
    secondary_y: bool = True,
) -> None:
    ymax = dataframe.iloc[:, 0].max()
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
                y0=ymax,
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
                y=ymax,
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
                y0=ymax,
                y1=0,
                type="rect",
                # xref="paper",
                # yref="paper",
                layer="below",
                fillcolor="LightSeaGreen",
            )

            figure.add_annotation(
                x=timestamp,
                y=ymax,
                text=name,
                textangle=90,
                # align="right",
                showarrow=False,
                xshift=10,
                yshift=-25,
            )

    figure.update_shapes(dict(xref="x", yref="y"))
    figure.update_xaxes(range=[dataframe.index.min(), dataframe.index.max()])
    figure.update_yaxes(rangemode="tozero")  # type="log",
    if secondary_y:
        figure.update_yaxes(title_text=attributes[0], secondary_y=False)
        figure.update_yaxes(title_text=attributes[1], secondary_y=True)
    figure.update_layout(
        title_text=title or "",
        template="simple_white",
        legend=dict(
            x=1,
            y=1,
            xanchor="right",
            yanchor="bottom",
            orientation="h",
        ),
    )

    figure.show()


# show residuals as kind of OHLC Charts
# attributes are tuple(truth, prediction) names of dataframe
def timeline_residuals_plot(
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
                # opacity=0.6,
                showlegend=False,
            ),
            Scatter(
                x=index,
                y=predictions,
                mode="lines",
                name="prediction",
                line=dict(color="lightblue", width=0.6, dash="dot"),
                showlegend=False,
                # opacity=0.6,
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
                increasing_line_color="lightseagreen",
                decreasing_line_color="lightsalmon",
            ),
        ]
    )

    figure.update_layout(
        title=title or "Model's Residuals",
        template="simple_white",
        xaxis_rangeslider_visible=True,
    )

    figure.show()


# zoomed plot on the predictions of two models and a reference
def timeline_predictions_interval_plot(
    dataframe: DataFrame,
    rnn_dataframe: DataFrame,
    baseline_dataframe: DataFrame,
) -> None:
    figure = Figure()
    value = Scatter(
        x=dataframe.index,
        y=dataframe["validations"],
        mode="lines",
        name="Reference",
        line=dict(color="rgba(0,0,0, 0.3)", width=1, dash="dot"),
    )

    figure.add_trace(value)
    baseline = Scatter(
        x=baseline_dataframe.index,
        y=baseline_dataframe.predictions,
        mode="lines",
        name="Linear Regression",
        opacity=0.8,
    )

    figure.add_trace(baseline)
    prediction = Scatter(
        x=rnn_dataframe.index,
        y=rnn_dataframe.predictions,
        mode="lines",
        name="NN",
        # marker=dict(),
        opacity=0.8,
        visible="legendonly",
    )

    figure.add_trace(prediction)
    figure.update_layout(
        showlegend=True,
        title_text="Predictions",
        template="simple_white",
        xaxis=dict(
            range=[
                rnn_dataframe.index.min(),
                rnn_dataframe.index.max(),
            ],
        ),
    )

    figure.update_xaxes(rangeslider_visible=True)
    figure.show()


def timeline_predictions_interval_with_staggings_plot(
    dataframe: DataFrame,
    staggered: DataFrame,
) -> None:
    figure = subplots.make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        specs=[
            [{"rowspan": 3}],
            [None],
            [{}],
            [{}],
        ],
        vertical_spacing=0.1,
    )

    prediction_plot = Scatter(
        x=dataframe.index,
        y=dataframe.predictions,
        mode="lines",
        name="predictions",
        # opacity=0.1,
        fill=None,
        showlegend=False,
        # line_color="gray",
        line=dict(color="gray", width=0.1),
        # hoverinfo="x+y",
        # stackgroup='one'
    )

    figure.add_trace(prediction_plot, row=1, col=1)
    staggered_plot = Scatter(
        x=staggered.index,
        y=staggered.predictions,
        mode="lines",
        name="staggered",
        # opacity=0.8,
        fill="tonexty",
        fillcolor="red",
        line=dict(color="gray", width=0.1),
        # hoverinfo="x+y",
        # stackgroup='one'
    )

    figure.add_trace(staggered_plot, row=1, col=1)
    residuals = (
        pd.merge(
            dataframe,
            staggered,
            how="outer",
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
    colors = [
        "lightseagreen" if c > 0 else "lightsalmon" for c in residuals["difference"]
    ]
    bar_plot = Bar(
        x=residuals.index,
        y=residuals.difference,
        name="difference",
        showlegend=False,
        marker_color=colors,
    )

    figure.add_trace(bar_plot, row=4, col=1)

    figure.update_xaxes(showticklabels=True, row=1, col=1)
    figure.update_xaxes(showticklabels=False, row=4, col=1, visible=False)
    figure.update_yaxes(
        title_text="difference",
        row=4,
        col=1,
        zeroline=True,
        zerolinecolor="gray",
    )

    figure.update_layout(
        showlegend=True,
        title_text="Predictions and Staggings",
        template="simple_white",
    )

    figure.show()
