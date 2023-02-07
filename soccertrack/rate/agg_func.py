import numpy as np


def get_agg_func(agg_func: str):
    """ Returns a function that aggregates the metrics matrix.
    
    Args:
        agg_func (str): Aggregation function. Can be one of the following:
        
            - None: No aggregation. Returns the metrics matrix.
            - "w_mean": Weighted mean. Returns the weighted mean of the metrics matrix.
            - "mean": Mean. Returns the mean of the metrics matrix.
            - "median": Median. Returns the median of the metrics matrix.
            - "max": Maximum. Returns the maximum of the metrics matrix.
            - "min": Minimum. Returns the minimum of the metrics matrix.
            
    Returns:
        function: Aggregation function.
    """
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


def get_time_series_agg(time_series_metrics: np.ndarray, num_agg_frame: int = 10):
    """ Returns the aggregated time series scores of the metrics matrix.
    
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
    return np.array(nframe_sum)


def get_time_series_agg_func(agg_func: str):
    """ Returns a function that aggregates the time series metrics.
    
    Args:
        agg_func (str): Aggregation function. Can be one of the following:
        
            - None: No aggregation. Returns the time series metrics.
            - "sum": Sum. Returns the sum of the time series metrics.
            - "mean": Mean. Returns the mean of the time series metrics.
            - "median": Median. Returns the median of the time series metrics.
            - "max": Maximum. Returns the maximum of the time series metrics.
            - "min": Minimum. Returns the minimum of the time series metrics.
            - "nframe_agg": Sum of aggregated time series metrics. Returns the aggregated time series metrics.
            - "nframe_diff_max": Returns the maximum of the difference of aggregated time series metrics.
            
    Returns:
        function: Aggregation function.
    """
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
    elif agg_func == "nframe_agg":
        return lambda time_series_metrics: get_time_series_agg(time_series_metrics)
    elif agg_func == "nframe_diff_max":
        return lambda time_series_metrics: np.max(np.diff(get_time_series_agg(time_series_metrics)))
    else:
        raise ValueError(f"Aggregation function {agg_func} not supported.")