from soccertrack.dataframe import CoordinatesDataFrame
import pandas as pd
import numpy as np
from soccertrack.rate.agg_func import get_agg_func

#setup initial variables

grid_xmin = -52.5
grid_ymin = -34
grid_xmax = 52.5
grid_ymax = 34
window_x = 16
window_y = 12

def grid_count(
    ball_traj: np.ndarray
    ) -> list[int]:
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

    pitch_length_x = np.linspace(grid_xmin, grid_xmax, window_x+1)
    pitch_length_y = np.linspace(grid_ymin, grid_ymax, window_y+1)
    
    moving_area_count = []

    for idx_x in range(1,window_x+1,1):
        for idx_y in range(1,window_y+1,1):
            moving_area = ball_traj[(ball_traj[:,0] >= pitch_length_x[idx_x-1]) & 
                                (ball_traj[:,0] <= pitch_length_x[idx_x]) & 
                                (ball_traj[:,1] >= pitch_length_y[idx_y-1]) & 
                                (ball_traj[:,1] <= pitch_length_y[idx_y])]
            moving_area_count.append(len(moving_area))
    return moving_area_count


#calulate xG
def rate_xG(codf: CoordinatesDataFrame, xg_mtx: pd.Series, agg_func="w_mean"):
    ball_traj = list(codf.iter_players())[-1][1].values
    moving_area_count = grid_count(ball_traj)
    moving_area_count = np.array(moving_area_count).reshape(window_x, window_y)

    xg_score = get_agg_func(agg_func)(xg_mtx, moving_area_count)
    return xg_score

#calulate xT
def rate_xT(codf: CoordinatesDataFrame, xt_mtx: pd.Series, agg_func="w_mean"):
    ball_traj = list(codf.iter_players())[-1][1].values
    moving_area_count = grid_count(ball_traj)
    moving_area_count = np.array(moving_area_count).reshape(window_x, window_y)

    xt_score = get_agg_func(agg_func)(xt_mtx, moving_area_count)
    return xt_score

