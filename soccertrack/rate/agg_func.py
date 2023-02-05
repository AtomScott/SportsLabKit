import numpy as np

def get_agg_func(agg_func):
    if agg_func is None:
        return lambda x: x
    elif agg_func == "w_mean":
            return lambda metrics_mtx, ma_count_arr: np.sum(metrics_mtx * ma_count_arr) / np.sum(ma_count_arr)
    elif agg_func == "mean":
        return lambda metrics_mtx, ma_count_arr: np.mean(metrics_mtx * ma_count_arr)
    elif agg_func == "median":
        return lambda metrics_mtx, ma_count_arr: np.median(metrics_mtx * ma_count_arr)
    elif agg_func == "max":
        return lambda metrics_mtx, ma_count_arr: np.max(metrics_mtx * ma_count_arr)
    elif agg_func == "min":
        return lambda metrics_mtx, ma_count_arr: np.min(metrics_mtx * ma_count_arr)
    else:
        raise ValueError(f"Aggregation function {agg_func} not supported.")