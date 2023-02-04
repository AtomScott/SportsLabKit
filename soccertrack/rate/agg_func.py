import numpy as np


def get_agg_func(agg_func):
    if agg_func is None:
        return lambda x: x
    elif agg_func == "mean":
        return np.mean
    elif agg_func == "median":
        return np.median
    elif agg_func == "max":
        return np.max
    elif agg_func == "min":
        return np.min
    else:
        raise ValueError(f"Aggregation function {agg_func} not supported.")
