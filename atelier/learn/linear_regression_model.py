from pandas import DataFrame
from sklearn.linear_model import LinearRegression


def predict(
    X_train: DataFrame,
    y_train: DataFrame,
    X_test: DataFrame,
    y_test: DataFrame,
) -> DataFrame:
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    dataframe = DataFrame(y_test)
    dataframe = dataframe.assign(prediction=prediction)
    # dataframe = dataframe.sort_index()
    return dataframe
