import numpy as np


def get_agg_func(agg_func):
    if agg_func is None:
        return lambda metrics_mtx, ma_count_arr: metrics_mtx * ma_count_arr
    elif agg_func == "w_mean":
        return lambda metrics_mtx, ma_count_arr: np.sum(
            metrics_mtx * ma_count_arr
        ) / np.sum(ma_count_arr)
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


def get_time_series_agg(time_series_metrics, num_agg_frame=10):
    nframe_sum = []
    start = 0
    end = num_agg_frame
    while end <= len(time_series_metrics):
        nframe_sum.append(sum(time_series_metrics[start:end]))
        start += 10
        end += 10
    nframe_sum.append(sum(time_series_metrics[start:]))
    return np.array(nframe_sum)


def get_time_series_agg_func(agg_func):
    if agg_func is None:
        return lambda time_series_metrics: time_series_metrics
    elif agg_func == "sum":
        return lambda time_series_metrics: np.sum(time_series_metrics)
    elif agg_func == "mean":
        return lambda time_series_metrics: np.mean(time_series_metrics)
    elif agg_func == "median":
        return lambda time_series_metrics: np.median(time_series_metrics)
    elif agg_func == "max":
        return lambda time_series_metrics: np.max(time_series_metrics)
    elif agg_func == "min":
        return lambda time_series_metrics: np.min(time_series_metrics)
    elif agg_func == "nframe_sum":
        return lambda time_series_metrics: get_time_series_agg(time_series_metrics)
    elif agg_func == "nframe_diff_max":
        return lambda time_series_metrics: np.max(
            np.diff(get_time_series_agg(time_series_metrics))
        )
    else:
        raise ValueError(f"Aggregation function {agg_func} not supported.")
