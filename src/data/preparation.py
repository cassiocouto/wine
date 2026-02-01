import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def bin_data(df, column, bins, labels):
    """
    Bin a continuous variable into categorical bins.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    column (str): The column name to be binned.
    bins (list): The bin edges.
    labels (list): The labels for the bins.

    Returns:
    pd.DataFrame: A new DataFrame with the binned column.
    """
    df_copy = df.copy()
    df_copy[column + "_binned"] = pd.cut(
        df_copy[column], bins=bins, labels=labels, include_lowest=True
    )
    return df_copy


def normalize_column(df, column):
    """
    Normalize a column in the dataframe using min-max scaling.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    column (str): The column name to be normalized.

    Returns:
    pd.DataFrame: A new DataFrame with the normalized column.
    """
    df_copy = df.copy()
    df_copy[column + "_normalized"] = (df_copy[column] - df_copy[column].min()) / (
        df_copy[column].max() - df_copy[column].min()
    )
    return df_copy


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all columns in the dataframe using min-max scaling.
    :param df: Description
    """
    df_copy = df.copy()
    for column in df_copy.columns:
        df_copy[column] = (df_copy[column] - df_copy[column].min()) / (
            df_copy[column].max() - df_copy[column].min()
        )
    return df_copy

def __check_test_size_percentage(test_size_percentage: float) -> None:
    if test_size_percentage is not None and (
        test_size_percentage <= 0 or test_size_percentage >= 1
    ):
        raise ValueError("test_size_percentage must be between 0 and 1.")

def prepare_data_for_classification(
    df: pd.DataFrame,
    target_column: str,
    random_state: int = 42,
    test_size_percentage: float = None,
) -> tuple:
    """
    Separates features and target variable from the dataframe.
    :param df: Description
    :param target_column: The name of the target column
    :param random_state: Random state for reproducibility
    :param test_size_percentage: Test size percentage
    :return: A tuple containing features (X) and target (y) as DataFrames/Series.
    """
    __check_test_size_percentage(test_size_percentage)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    if test_size_percentage is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size_percentage,
            random_state=random_state,
            shuffle=True,
        )
        return X_train, X_test, y_train, y_test
    return X, y


def prepare_data_for_classification_as_np_arrays(
    df: pd.DataFrame, target_column: str,
    random_state: int = 42,
    test_size_percentage: float = None
) -> tuple:
    """
    Separates features and target variable from the dataframe and converts them to NumPy arrays.
    :param df: Description
    :param target_column: The name of the target column
    """
    __check_test_size_percentage(test_size_percentage)
    X = df.drop(columns=[target_column]).to_numpy()
    y = df[target_column].to_numpy()
    if test_size_percentage is not None:
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        test_size = int(len(X) * test_size_percentage)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        return X_train, X_test, y_train, y_test
    
    return X, y
