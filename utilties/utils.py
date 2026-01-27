#
# print(unzip("Polynomial_Regression_and_Data_Transformation.zip", "."))
import numpy as np, pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from IPython.display import display


def removing_statistical_outliers(df, columns, quantile=None):
    """
    A function that removes outliers in the dataframe
    @param: df -> DataFrame
    @param: columns-> Specified columns
    @param: quantile-> if not specified 75% is taken by default
    @returns: new dataframe with outliers removed
    """
    if not quantile:
        quantile = 0.75
    for col in columns:
        Q1 = df[col].quantile(quantile)
        Q3 = df[col].quantile(1 - quantile)
        IQR = Q3 - Q1
        # print(f"Original Shape: {grouped_df.shape}")
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        # print(f"After Removing Outliers: {grouped_df.shape}")
    return df


def unique_values_percentage(df, columns):
    """
    A function to calculate the summary of categorical columns of a dataframe
        @param: df -> DataFrame
        @param: columns: list of categorical columns in the dataframe
        @returns: dictionary of summary of unique value counts and their percentages
    >>> x = pd.DataFrame({'temp':['hot','hot','cool']})

    """
    results = {}
    for column in columns:
        # Count unique values and calculate percentages
        value_counts = df[column].value_counts()
        percentages = df[column].value_counts(normalize=True) * 100

        # Combine counts and percentages into a DataFrame
        unique_summary = pd.DataFrame(
            {"column": column, "Count": value_counts, "Percentage (%)": percentages}
        )

        # Add the result to the dictionary
        results[column] = unique_summary

    return results


def goodness_of_fit(y_true, predictions):
    """
    @param: y_true: actual
    @param prediction: predicted
    @return:  a tuple of r2_score, sum of squared residuals,mean squared error,root mean squared error
    """
    accuracy = r2_score(y_true, predictions)
    # RSS
    rss = np.sum(np.square(y_true - predictions))
    # MSE
    mse = mean_squared_error(y_true=y_true, y_pred=predictions)
    # RMSE
    rmse = np.sqrt(mse)
    out = [("Accuracy:", accuracy), ("RSS:", rss), ("MSE:", mse), ("RMSE:", rmse)]

    return out


def memo(func):
    cache = {}

    def inner(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return inner


def variance_inflation(X_train, selected_features):
    vif = pd.DataFrame()
    vif["Features"] = X_train[selected_features].columns
    vif["VIF"] = [
        variance_inflation_factor(X_train[selected_features].values, i)
        for i in range(X_train[selected_features].shape[1])
    ]
    vif["VIF"] = round(vif["VIF"], 2)
    vif = vif.sort_values(by="VIF", ascending=False)
    return vif


def display_info(df):
    display(df)
    print("\n")
    display(df.info())
