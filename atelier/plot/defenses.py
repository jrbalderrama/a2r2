import plotly.express as px
import plotly.figure_factory as ff
from pandas import DataFrame
from plotly.graph_objs import Scatter


def facet_plot(
    dataframe: DataFrame,
    size: int,
    row: str,
    col: str,
) -> None:
    dataset = dataframe.query(f"n=={size}").reset_index()
    figure = px.line(
        dataset,
        x="departure_time",
        y="fpa",
        facet_row=row,
        facet_col=col,
        labels={"departure_time": "", "fpa": ""},
        # facet_row_spacing=0.01,
        # facet_col_spacing=0.01,
    )

    figure.update_yaxes(matches=None, showticklabels=False)
    figure.update_xaxes(showticklabels=False)
    # figure.update_coloraxes(showscale=False)

    trace = Scatter(
        x=dataset.departure_time,
        y=dataset.validations,
        name="validations",
        line=dict(color="gray", width=0.1, dash="dot"),
        opacity=0.35,
    )

    trace.update(showlegend=False)

    # draw same curve in all element of the grid
    for i, _ in enumerate(dataset[row].unique(), start=1):
        for j, _ in enumerate(dataset[col].unique(), start=1):
            figure.add_trace(trace, row=i, col=j)

    figure.update_layout(
        template="plotly_white",
        title=f"FPA for n={size}",
        xaxis_title="date",
        yaxis_title="validations",
    )

    figure.show()


def distributions_plot(
    dataframe: DataFrame,
    *,
    curve_type="normal",
) -> None:
    dataframe_ = dataframe.copy()
    dataframe_["noise"] = dataframe_["validations"] - dataframe_["fpa"]
    figure = ff.create_distplot(
        [dataframe_[c] for c in dataframe_.columns],
        dataframe_.columns,
        curve_type=curve_type,
        # bin_size=[3, 3, 3],
    )

    figure.show()
