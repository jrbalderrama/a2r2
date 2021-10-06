from pandas import DataFrame
from plotly.graph_objs import Figure, Scatter


# show a timeseries graph of a selected attribute
def timeline_plot(
    dataframe: DataFrame,
    attribute: str,
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
        title_text=attribute,
        template="simple_white",
    )

    figure.update_xaxes(showgrid=True)
    figure.show()
