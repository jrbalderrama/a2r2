from typing import Optional, Sequence, Tuple

from pandas import DataFrame, Timestamp
from plotly import subplots
from plotly.graph_objs import Figure, Scatter


# show a timeseries graph of a selected attribute
def timeline_plot(
    dataframe: DataFrame,
    attribute: str,
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


# TODO rework
# show timeline divided bt delimiters and holidays
def timeline_plot_with_annotations(
    dataframe: DataFrame,
    attributes: Sequence[str],
    delimiters: Sequence[Timestamp],
    holidays: Tuple[Timestamp, Timestamp],
) -> None:
    dmin = dataframe["nombre_etudiant"].values.min()
    dmax = dataframe["nombre_etudiant"].values.max()
    figure = subplots.make_subplots(specs=[[{"secondary_y": True}]])
    for counter, column in enumerate(attributes):
        secondary_y = False if counter % 2 == 0 else True
        scatter = Scatter(
            x=dataframe.index,
            y=dataframe[column],
            mode="lines",
            name=column,
        )

        figure.add_trace(
            scatter,
            secondary_y=secondary_y,
        )

    for delimiter in delimiters:
        figure.add_shape(
            type="line",
            x0=delimiter,
            x1=delimiter,
            y0=dmax,
            y1=0,
            line=dict(
                # color="Gray",
                width=1,
                dash="dashdot",
            ),
        )

    figure.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        layer="below",
        fillcolor="LightSeaGreen",
        x0=holidays[0],
        x1=holidays[1],
        y0=dmax,
        y1=0,
    )

    figure.add_annotation(
        x=holidays[0],
        y=dmax,
        align="right",
        text="holidays",
        showarrow=False,
        yshift=-25,
        textangle=90,
        xshift=10,
    )

    figure.add_annotation(
        x=delimiters[0],
        y=dmax,
        text="validation",
        showarrow=True,
        yshift=-15,
    )

    figure.add_annotation(
        x=delimiters[1],
        y=dmax,
        text="test",
        showarrow=True,
    )

    figure.update_shapes(dict(xref="x", yref="y"))
    figure.update_yaxes(
        rangemode="tozero",
        # type="log",
    )

    figure.update_xaxes(range=[dataframe.index.min(), dataframe.index.max()])
    figure.update_yaxes(title_text=attributes[0], secondary_y=False)
    figure.update_yaxes(title_text=attributes[1], secondary_y=True)
    figure.update_layout(
        title_text="Count of Buses & Classes",
        template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    figure.show()
