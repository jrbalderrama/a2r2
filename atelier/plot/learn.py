from plotly.graph_objs import Figure, Scatter


# TODO use a dataframe
def losses_plot(
    train_losses,
    val_losses,
) -> None:
    figure = Figure()
    tics = [*range(len(train_losses) + 1)]
    value = Scatter(
        x=tics,
        y=train_losses,
        mode="lines",
        name="Training",
        marker=dict(),
    )

    figure.add_trace(value)
    value = Scatter(
        x=tics,
        y=val_losses,
        mode="lines",
        name="Validation",
        marker=dict(),
    )

    figure.add_trace(value)
    figure.update_layout(title_text="Losses")
    figure.update_xaxes(title_text="epoch")
    figure.update_yaxes(title_text="loss (%)")
    figure.show()
