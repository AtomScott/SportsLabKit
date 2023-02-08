from functools import partial
from typing import Optional

import numpy as np


def get_agg_func(agg_func: Optional[str] = None, **agg_kwargs):
    if agg_func is None:
        return lambda x: x

    elif agg_func == "w_mean":
        return lambda x: np.average(x, weights=agg_kwargs.get("weights", None))
    elif agg_func == "mean":
        return lambda x: np.mean(x)
    elif agg_func == "median":
        return lambda x: np.median(x)
    elif agg_func == "max":
        return lambda x: np.max(x)
    elif agg_func == "min":
        return lambda x: np.min(x)
    elif agg_func == "sum":
        return lambda x: np.sum(x)
    elif agg_func == "std":
        return lambda x: np.std(x)
    elif agg_func == "var":
        return lambda x: np.var(x)
    elif agg_func == "nframe_diff_max":
        return partial(
            nframe_diff_max, num_agg_frame=agg_kwargs.get("num_agg_frame", 10)
        )
    elif callable(agg_func):
        return agg_func
    else:
        raise ValueError(f"agg_func {agg_func} is not supported.")


def nframe_diff_max(time_series_metrics: np.ndarray, num_agg_frame: int = 10):
    """Returns the aggregated time series scores of the metrics matrix.

    Args:
        time_series_metrics (np.array): Time series scores of the metrics matrix.
        num_agg_frame (int): Number of frames to aggregate. default: 10.

    Returns:
        nframe_sum (np.array): Aggregated time series scores.
    """
    nframe_sum = []
    start = 0
    end = num_agg_frame
    while end <= len(time_series_metrics):
        nframe_sum.append(sum(time_series_metrics[start:end]))
        start += 10
        end += 10
    nframe_sum.append(sum(time_series_metrics[start:]))
    nframe_diff_max = np.max(np.diff(nframe_sum))
    return nframe_diff_max
