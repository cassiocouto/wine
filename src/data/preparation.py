import pandas as pd
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
    df_copy[column + '_binned'] = pd.cut(df_copy[column], bins=bins, labels=labels, include_lowest=True)
    return df_copy