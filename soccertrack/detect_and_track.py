"""Implementation of object detection and tracking, and plotting of the output to pitch coordinates."""

import os
# import sys

import numpy as np
from rich import print
# from soccertrack.utils import load_config
# from soccertrack.utils.camera import load_cameras, find_intrinsic_camera_parameters
from soccertrack.detect import detect_objects
from typing import List, Dict, Any, TypeVar, Callable, Tuple
# from soccertrack.utils import cv2pil

from mplsoccer import Pitch
import matplotlib.pyplot as plt
from IPython.display import Video, display


# cfg = load_config('./detect_and_track.yaml')
# cameras = load_cameras(cfg['cameras'])


# """
# TODO: detection checklist
#     - return results as list of Detection Objects -> OK
#     - option to visualize detections? 
#     - aggregate detections from multiple cameras? 
#     - drop detections that are out of bounds -> OK
#     - Are detection results visualized and saved?
# """

F = TypeVar('F', bound=Callable[..., Any])


def merge_dict_of_lists(
    d1:dict, 
    d2:dict
) -> Dict[dict, dict]:
    """Merge two dictionaries of lists.

    Args:  
        d1 (dict): Dictionary of lists.
        d2 (dict): Dictionary of lists.

    Returns:
        ret: Dictionary of lists.
    """
    ret = {k:v for k,v in d1.items()}
    for k in d1.items():
        if k in d2.keys():
            ret[k] += d2[k]
        else:
            ret[k] = d2[k]
    return ret

def display_detected_video(cfg, cameras):
    """Display detected video.
    
    Args:
        cfg (dict): Configuration dictionary.
        cameras (list): List of cameras(camera configuration).
    
    Returns:
        video (Video): Video object.
        ball_detections (list): List of ball detections.
        player_detections (list): List of player detections.
    """

    player_detections, ball_detections = detect_objects(
        cameras,
        model_name='yolov5s',
        size=3000,
        batch_size=8,
        filter_range=False
    )

    print("N ball detections:", sum(len(x) for x in ball_detections.values()))
    print("N player detections:", sum(len(x) for x in player_detections.values()))

    all_candidate_detection = merge_dict_of_lists(ball_detections, player_detections)

    for i, camera in enumerate(cameras):
        save_path = os.path.join(cfg.outdir, f'camera[{i}]-visualize_candidate_detections.mp4')
        camera.visualize_candidate_detections(
            candidate_detections=all_candidate_detection, 
            filter_range=False,
            save_path=save_path
            )
        video = Video(save_path, width=600)
        display(video)
    return video, ball_detections, player_detections


def display_detected_video_with_pitch(
    player_detections: dict, 
    cameras: list
):
    """Display detected video with pitch overlay.

    Args:
        player_detections (dict): Dictionary of lists of player detections.
        cameras (list): List of cameras.
    
    Returns:    
        pitch (Pitch): Pitch object.
        xs (list): List of x coordinates of detected players.
        ys (list): List of y coordinates of detected players.
        kxs (list): List of x coordinates of detected pitch keypoint.
        kys (list): List of y coordinates of detected pitch keypoint.
    """

    xs,ys =[],[]
    for pd in player_detections[0]:
        if pd.camera.label == cameras[0].label:
            px, py = cameras[0].video2pitch(np.array([pd._x, pd._y])).squeeze()
        elif pd.camera.label == cameras[1].label:
            px, py = cameras[1].video2pitch(np.array([pd._x, pd._y])).squeeze()
        else:
            raise ValueError("camera label not found")
        xs.append(px)
        ys.append(py)

    kxs,kys =[],[]
    for camera in cameras:
        for x, y in camera.source_keypoints:
            x, y = camera.video2pitch(np.array([x, y])).squeeze()
            kxs.append(x)
            kys.append(y)

    pitch = Pitch(pitch_color='black', line_color=(.3,.3,.3), pitch_type='custom', pitch_length=105, pitch_width=68) 
    # fig, ax = pitch.draw()
    plt.scatter(xs, ys, color='deeppink')
    plt.scatter(kxs, kys, color='red')
    plt.show()

    return pitch, xs, ys, kxs, kys
