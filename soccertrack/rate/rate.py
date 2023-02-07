import numpy as np
import pandas as pd

from soccertrack.dataframe import CoordinatesDataFrame
from soccertrack.rate.agg_func import get_agg_func, get_time_series_agg_func

def grid_count(ball_traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Divides the trajectory of a ball into a grid and returns a list of moving areas and corresponding counts.

    Args:
        ball_traj(np.ndarray): A 2D numpy array representing the trajectory of a ball.

    Returns:
        moving_area_count(np.ndarray): A 2D numpy array representing the count of elements in each moving area.
        moving_area_indices(np.ndarray): A 1D numpy array representing the indices of the moving area for each element of `ball_traj`.
    """
    # setup initial variables
    grid_xmin = 0
    grid_ymin = 0
    grid_xmax = 105
    grid_ymax = 68
    window_x = 16
    window_y = 12

    pitch_length_x = np.linspace(grid_xmin, grid_xmax, window_x + 1)
    pitch_length_y = np.linspace(grid_ymin, grid_ymax, window_y + 1)

    moving_area_count, _, _ = np.histogram2d(ball_traj[:,0], ball_traj[:,1], bins=(pitch_length_x, pitch_length_y))
    moving_area_count = moving_area_count.astype(int)

    moving_area_indices = np.zeros(len(ball_traj), dtype=float)
    moving_area_indices.fill(np.nan)
    for idx, coord in enumerate(ball_traj):
        x_bin = np.digitize(coord[0], pitch_length_x) - 1
        y_bin = np.digitize(coord[1], pitch_length_y) - 1
        if x_bin >= 0 and x_bin < window_x and y_bin >= 0 and y_bin < window_y:
            moving_area_indices[idx] = int(x_bin * window_y + y_bin)

    return moving_area_count, moving_area_indices


# calulate xG
def rate_xG(codf: CoordinatesDataFrame, agg_func="w_mean"):
    """calculate xG score for a given CoordinatesDataFrame.
    
    Args:
        codf (CoordinatesDataFrame): A CoordinatesDataFrame object.
        agg_func (str, optional): Aggregation function. Defaults to "w_mean".
    
    Returns:
        xg_score (float): xG score.
    """
    ball_traj = list(codf.iter_players())[-1][1].values
    moving_area_count, _ = grid_count(ball_traj)

    xg_score = get_agg_func(agg_func)(xg_mtx, moving_area_count)
    return xg_score

def rate_xG_time_series(codf: CoordinatesDataFrame, agg_func="nframe_diff_max"):
    """calculate Time-Series xG score for a given CoordinatesDataFrame.
    
    Args:
        codf (CoordinatesDataFrame): A CoordinatesDataFrame object.
        agg_func (str, optional): Aggregation function. Defaults to "nframe_diff_max".
    
    Returns:
        xg_score (float): Time-Series xG score.
    """
    ball_traj = list(codf.iter_players())[-1][1].values
    _, moving_area_indices = grid_count(ball_traj)
    
    xg_mtx_flatten = xg_mtx.flatten()
    xg_score_per_frame = np.zeros(len(moving_area_indices))
    for idx, row in enumerate(moving_area_indices):
        if np.isnan(row):
            xg_score_per_frame[idx] = 0
        else:
            xg_score_per_frame[idx] = xg_mtx_flatten[int(row)]
    xg_score = get_time_series_agg_func(agg_func)(xg_score_per_frame)
    return xg_score

# calulate xT
def rate_xT(codf: CoordinatesDataFrame, agg_func="w_mean"):
    """calculate xT score for a given CoordinatesDataFrame.
    
    Args:
        codf (CoordinatesDataFrame): A CoordinatesDataFrame object.
        agg_func (str, optional): Aggregation function. Defaults to "w_mean".
    
    Returns:
        xt_score (float): xT score.
    """
    ball_traj = list(codf.iter_players())[-1][1].values
    moving_area_count, _ = grid_count(ball_traj)

    xt_score = get_agg_func(agg_func)(xt_mtx, moving_area_count)
    return xt_score

def rate_xT_time_series(codf: CoordinatesDataFrame, agg_func="nframe_diff_max"):
    """calculate Time-Series xT score for a given CoordinatesDataFrame.
    
    Args:
        codf (CoordinatesDataFrame): A CoordinatesDataFrame object.
        agg_func (str, optional): Aggregation function. Defaults to "nframe_diff_max".
    
    Returns:
        xt_score (float): Time-Series xT score.
    """
    ball_traj = list(codf.iter_players())[-1][1].values
    _, moving_area_indices = grid_count(ball_traj)
    
    xt_mtx_flatten = xt_mtx.flatten()
    xt_score_per_frame = np.zeros(len(moving_area_indices))
    for idx, row in enumerate(moving_area_indices):
        if np.isnan(row):
            xt_score_per_frame[idx] = 0
        else:
            xt_score_per_frame[idx] = xt_mtx_flatten[int(row)]
    xt_score = get_time_series_agg_func(agg_func)(xt_score_per_frame)
    return xt_score


xg_mtx = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        [0.11, 0.11, 0.07, 0.07, 0.07, 0.15, 0.15, 0.07, 0.07, 0.07, 0.11, 0.11],
        [0.11, 0.11, 0.07, 0.07, 0.07, 0.15, 0.15, 0.07, 0.07, 0.07, 0.11, 0.11],
        [0.11, 0.11, 0.07, 0.07, 0.1, 0.34, 0.34, 0.1, 0.07, 0.07, 0.11, 0.11],
    ]
)


xt_mtx = np.array(
    [
        [
            0.002,
            0.003,
            0.003,
            0.004,
            0.005,
            0.004,
            0.004,
            0.005,
            0.004,
            0.003,
            0.003,
            0.003,
        ],
        [
            0.003,
            0.003,
            0.004,
            0.005,
            0.006,
            0.005,
            0.005,
            0.005,
            0.004,
            0.003,
            0.005,
            0.004,
        ],
        [
            0.004,
            0.004,
            0.004,
            0.005,
            0.007,
            0.007,
            0.007,
            0.006,
            0.004,
            0.003,
            0.004,
            0.003,
        ],
        [
            0.004,
            0.004,
            0.004,
            0.004,
            0.005,
            0.005,
            0.004,
            0.004,
            0.004,
            0.004,
            0.004,
            0.004,
        ],
        [
            0.005,
            0.005,
            0.005,
            0.004,
            0.005,
            0.005,
            0.004,
            0.004,
            0.004,
            0.004,
            0.004,
            0.004,
        ],
        [
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
            0.005,
            0.006,
            0.006,
            0.006,
            0.006,
            0.005,
            0.006,
        ],
        [
            0.007,
            0.008,
            0.008,
            0.008,
            0.008,
            0.008,
            0.008,
            0.007,
            0.007,
            0.007,
            0.007,
            0.007,
        ],
        [
            0.008,
            0.01,
            0.01,
            0.009,
            0.009,
            0.01,
            0.009,
            0.01,
            0.009,
            0.009,
            0.008,
            0.008,
        ],
        [
            0.011,
            0.012,
            0.013,
            0.013,
            0.014,
            0.012,
            0.013,
            0.013,
            0.012,
            0.012,
            0.011,
            0.01,
        ],
        [
            0.014,
            0.015,
            0.016,
            0.017,
            0.018,
            0.019,
            0.017,
            0.017,
            0.016,
            0.016,
            0.015,
            0.013,
        ],
        [
            0.019,
            0.022,
            0.024,
            0.024,
            0.024,
            0.025,
            0.025,
            0.025,
            0.023,
            0.022,
            0.02,
            0.018,
        ],
        [
            0.026,
            0.031,
            0.036,
            0.036,
            0.037,
            0.035,
            0.033,
            0.033,
            0.034,
            0.034,
            0.029,
            0.025,
        ],
        [
            0.039,
            0.044,
            0.049,
            0.047,
            0.046,
            0.042,
            0.052,
            0.048,
            0.044,
            0.046,
            0.042,
            0.037,
        ],
        [
            0.051,
            0.06,
            0.068,
            0.065,
            0.077,
            0.116,
            0.089,
            0.068,
            0.068,
            0.067,
            0.056,
            0.048,
        ],
        [
            0.063,
            0.077,
            0.096,
            0.096,
            0.124,
            0.176,
            0.171,
            0.126,
            0.089,
            0.094,
            0.075,
            0.06,
        ],
        [
            0.068,
            0.094,
            0.136,
            0.126,
            0.158,
            0.371,
            0.413,
            0.158,
            0.123,
            0.103,
            0.092,
            0.066,
        ],
    ]
)