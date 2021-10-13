from typing import Dict

import numpy as np
from pandas import DataFrame
from sklearn import metrics, model_selection
from sklearn.model_selection import TimeSeriesSplit

DEFAULT_REGRESSION_SCORING = {
    "r2": "Score R2",
    "max_error": "Max Error",
    "explained_variance": "Explained Variance",
    "neg_mean_absolute_error": "Mean Absolute Error",
    "neg_root_mean_squared_error": "Root Mean Squared Error",
}


# Evaluate a 'time series' regression model
def timeline_evaluate_regression(
    model,
    X,
    y,
    *,
    scoring: Dict[str, str] = DEFAULT_REGRESSION_SCORING,
    splits: int = 3,
    max_train_size: int = None,
    test_size: int = None,
    gap: int = 0,
) -> None:
    """
    # test the model: not very accurate (?) during cross_validation
    >>> timeline_evaluate_regression(
    >>>     regressor_pipeline.steps[-1][1],  # last element of pipeline
    >>>     X,
    >>>     y.to_numpy().ravel(),  # if regressor requires 1d array
    >>> )
    """
    cv_generator = TimeSeriesSplit(
        n_splits=splits,
        max_train_size=max_train_size,
        test_size=test_size,
        gap=gap,
    )

    results = model_selection.cross_validate(
        model,
        X,
        y,
        cv=cv_generator,
        scoring=[*scoring.keys()],
        n_jobs=-1,
    )

    for k, v in results.items():
        if k.startswith("test_"):
            mean = np.abs(v).mean()
            std = np.abs(v).std()
            print(f"{scoring[k[5:]]}: {mean:.3f} +/- {std:.3f}")


def print_metrics(
    dataframe: DataFrame,
    reference: str = "references",
    prediction: str = "predictions",
) -> None:
    metrics_ = {
        "mae": metrics.mean_absolute_error(
            dataframe[reference],
            dataframe[prediction],
        ),
        "rmse": metrics.mean_squared_error(
            dataframe[reference],
            dataframe[prediction],
        )
        ** 0.5,
        "r2": metrics.r2_score(
            dataframe[reference],
            dataframe[prediction],
        ),
    }

    print(
        f"\tMean Absolute Error:     {metrics_['mae']:.3f}\n"
        f"\tRoot Mean Squared Error: {metrics_['rmse']:.3f}\n"
        f"\tR^2 Score:               {metrics_['r2']:.3f}"
    )
