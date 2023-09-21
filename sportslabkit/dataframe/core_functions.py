from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

import pandas as pd


if TYPE_CHECKING:
    from .slk_dataframe import SLKDataFrame



def required_columns(required_columns: list[str]) -> callable:
    """Decorator to check if DataFrame has required columns.

    Args:
        required_columns (list[str]): List of required column names.

    Returns:
        callable: Wrapped function.
    """
    def decorator(func: callable) -> callable:
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args: object, **kwargs: object) -> object:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns for this operation: {missing_columns}")
            return func(df, *args, **kwargs)
        return wrapper
    return decorator

@required_columns(['bb_left', 'bb_top', 'bb_width', 'bb_height'])
def visualize_boxes(df: 'SLKDataFrame') -> None:
    """Visualize bounding boxes.

    Args:
        df (pd.DataFrame): DataFrame containing bounding box columns.

    Returns:
        None
    """
    print("Visualizing boxes (core)...")
    # Actual implementation here

@required_columns(['x', 'y'])
def visualize_points(df: 'SLKDataFrame') -> None:
    """Visualize points.

    Args:
        df (pd.DataFrame): DataFrame containing point columns.

    Returns:
        None
    """
    print("Visualizing points (core)...")
    # Actual implementation here
