import functools
from typing import Callable, Optional, Tuple

import numpy as np
from pandas import DataFrame
from sklearn import feature_selection, pipeline
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

# scaler = MinMaxScaler()  # RobustScaler()  # StandardScaler()  # MinMaxScaler()
# loss_fn = MSELoss()  # L1Loss()
# optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
# runner = rnnmodel.RunnerHelper(model=model, loss_fn=loss_fn, optimizer=optimizer)


# runner.train(train_loader, val_loader, n_epochs=EPOCHS)
# plot.losses_plot(*runner.get_losses())
# predictions, values = runner.evaluate(test_loader)
# lstm_result = inverse_transform(values, predictions, X_test.index, scaler)


# NOTE: MLPRegressor fixes the loss function to MSE (cannot be changed)
def get_default_regressor(
    hidden_layer_sizes: Tuple[int, ...] = (64, 64, 64),
    batch_size: int = 64,
    max_iterations: int = 256,
    random_state: Optional[int] = None,
) -> MLPRegressor:
    # linear_model.RANSACRegressor()
    # linear_model.HuberRegressor(max_iter=500)
    # linear_model.LassoLars(alpha=2, normalize=False)
    # linear_model.SGDRegressor()
    # linear_model.TheilSenRegressor()
    # linear_model.LinearRegression()
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        batch_size=batch_size,
        max_iter=max_iterations,
        random_state=random_state,
        shuffle=False,
        # verbose=True,
        # tol=1e-6,
        # n_iter_no_change=64,
        # alpha=1e-6,#,0.0001,
    )


def make_pipeline(
    transformer: ColumnTransformer,
    # NOTE: this should be 'regressor: Intersection[BaseEstimator, RegressorMixin]'
    regressor: RegressorMixin,
    *,
    k: int = 0,
    score_function: Optional[Callable] = None,
    random_state: Optional[int] = None,
) -> Pipeline:
    if not score_function:
        score_function = functools.partial(
            feature_selection.mutual_info_regression,
            random_state=random_state,
        )

    return (
        pipeline.make_pipeline(
            transformer,
            SelectKBest(score_func=score_function, k=k),
            regressor,
        )
        if k > 0
        else pipeline.make_pipeline(
            transformer,
            regressor,
        )
    )


# setup a dataframe with results
def predict_and_compare(
    pipeline: Pipeline,
    X: DataFrame,
    y: DataFrame,
) -> DataFrame:
    results = DataFrame(
        pipeline.predict(X).astype(np.int_),
        index=y.index,
        columns=["predictions"],
    ).assign(references=y)

    results["residuals"] = results["references"] - results["predictions"]

    return results
