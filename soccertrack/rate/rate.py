import numpy as np
import pandas as pd

from soccertrack.dataframe import CoordinatesDataFrame
from soccertrack.rate.agg_func import get_agg_func

# setup initial variables

grid_xmin = 0
grid_ymin = 0
grid_xmax = 105
grid_ymax = 68
window_x = 16
window_y = 12


def grid_count(ball_traj: np.ndarray) -> list[int]:
    """
    Divides the trajectory of a ball into a grid and returns a list of moving areas and corresponding counts.

    Args:
        ball_traj(np.ndarray): A 2D numpy array representing the trajectory of a ball.
        grid_xmin(float): The minimum x value of the grid.
        grid_ymin(float): The minimum y value of the grid.
        grid_xmax(float): The maximum x value of the grid.
        grid_ymax(float): The maximum y value of the grid.
        window_x(int): The number of divisions along the x axis.
        window_y(int): The number of divisions along the y axis.

    Returns:
    moving_area_count(list[int]): A tuple of two lists, the first of which is a list of moving areas (each of which is a 2D numpy array) and the second of which is a corresponding list of counts of the number of elements in each moving area.
    """

    pitch_length_x = np.linspace(grid_xmin, grid_xmax, window_x + 1)
    pitch_length_y = np.linspace(grid_ymin, grid_ymax, window_y + 1)

    moving_area_count = []

    for idx_x in range(1, window_x + 1, 1):
        for idx_y in range(1, window_y + 1, 1):
            moving_area = ball_traj[
                (ball_traj[:, 0] >= pitch_length_x[idx_x - 1])
                & (ball_traj[:, 0] <= pitch_length_x[idx_x])
                & (ball_traj[:, 1] >= pitch_length_y[idx_y - 1])
                & (ball_traj[:, 1] <= pitch_length_y[idx_y])
            ]
            moving_area_count.append(len(moving_area))
    return moving_area_count


# calulate xG
def rate_xG(codf: CoordinatesDataFrame, agg_func="w_mean"):
    ball_traj = list(codf.iter_players())[-1][1].values
    moving_area_count = grid_count(ball_traj)
    moving_area_count = np.array(moving_area_count).reshape(window_x, window_y)

    xg_score = get_agg_func(agg_func)(xg_mtx, moving_area_count)
    return xg_score


# calulate xT
def rate_xT(codf: CoordinatesDataFrame, agg_func="w_mean"):
    ball_traj = list(codf.iter_players())[-1][1].values
    moving_area_count = grid_count(ball_traj)
    moving_area_count = np.array(moving_area_count).reshape(window_x, window_y)

    xt_score = get_agg_func(agg_func)(xt_mtx, moving_area_count)
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
