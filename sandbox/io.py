from ast import literal_eval
from typing import Any, Union

import dateutil.parser
import numpy as np
import pandas as pd
from pandas._typing import FilePath, WriteBuffer


def auto_string_parser(value: str) -> Any:
    """Auxiliary function to parse string values.

    Args:
        value (str): String value to parse.

    Returns:
        value (any): Parsed string value.
    """
    # automatically parse values to correct type
    if value.isdigit():
        return int(value)
    if value.replace(".", "", 1).isdigit():
        return float(value)
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "nan":
        return np.nan
    if value.lower() == "inf":
        return np.inf
    if value.lower() == "-inf":
        return -np.inf

    try:
        return literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    try:
        return dateutil.parser.parse(value)
    except (ValueError, TypeError):
        pass
    return value


def save_dataframe(
    df: pd.DataFrame,
    path_or_buf: Union[FilePath, WriteBuffer[bytes], WriteBuffer[str]],
) -> None:
    """Save a dataframe to a file.

    Args:
        df (pd.DataFrame): Dataframe to save.
        path_or_buf (FilePath | WriteBuffer[bytes] | WriteBuffer[str]): Path to save the dataframe.
    """
    if df.attrs:
        # write dataframe attributes to the csv file
        with open(path_or_buf, "w") as f:
            for k, v in df.attrs.items():
                f.write(f"#{k}:{v}\n")
    df.to_csv(path_or_buf, mode="a")


def load_dataframe(
    path_or_buf: Union[FilePath, WriteBuffer[bytes], WriteBuffer[str]],
) -> pd.DataFrame:
    """Load a dataframe from a file.

    Args:
        path_or_buf (FilePath | WriteBuffer[bytes] | WriteBuffer[str]): Path to load the dataframe.

    Returns:
        df (pd.DataFrame): Dataframe loaded from the file.
    """
    attrs = {}
    with open(path_or_buf, "r") as f:
        for line in f:
            if line.startswith("#"):
                k, v = line[1:].strip().split(":")
                attrs[k] = auto_string_parser(v)
            else:
                break

    skiprows = len(attrs)
    df = pd.read_csv(path_or_buf, header=[0, 1, 2], index_col=0, skiprows=skiprows)
    df.attrs = attrs
    return df
