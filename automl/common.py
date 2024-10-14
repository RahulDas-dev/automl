from typing import Iterable, List

import pandas as pd


def as_list(val: str | List | Iterable) -> List:
    """Ensure the input value is converted into a list."""
    if isinstance(val, str):
        return [val]

    if hasattr(val, "__iter__"):
        return list(val)

    return val


def check_X_for_type(X):
    """Checks if input of the Selector is of the required dtype"""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Provided variable X is not of type pandas.DataFrame")
    if X.empty:
        raise ValueError("Provided variable X is an empty DataFrame")


def check_column_length(columns: List[str]):
    """Check if no column is selected"""
    if len(columns) == 0:
        raise ValueError("Expected columns to be at least of length 1, found length of 0 instead")
