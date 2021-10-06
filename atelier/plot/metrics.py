import plotly.express as px
from pandas import DataFrame


# show the anonymity set of a dataframe as a barplot
def anonymity_sets_plot(
    dataframe: DataFrame,
) -> None:
    figure = px.bar(
        dataframe,
        x="cardinality",
        y="occurrences",
        color="occurrences",
        color_continuous_scale="Bluered",
        title="Anonymity Sets",
    )

    figure.update_coloraxes(showscale=False)
    figure.show()


# show the entropies as a dataframe as barplot
def entropies_plot(
    dataframe: DataFrame,
) -> None:
    figure = px.bar(
        dataframe,
        x="entropy",
        y="attribute",
        orientation="h",
        color="attribute",
        title="Entropies",
    )

    figure.update_traces(
        texttemplate="%{x:.2f}",
        textposition="auto",
    )

    figure.update_layout(showlegend=False)
    figure.show()
