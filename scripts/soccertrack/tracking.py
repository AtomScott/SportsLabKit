import json
import importlib

from pathlib import Path
from typing import Optional
import soccertrack
from soccertrack import Camera, detection_model
from soccertrack.image_model import TorchReIDModel
from soccertrack.logger import set_log_level
from soccertrack.metrics import (
    CosineCMM,
    EuclideanCMM,
    hota_score,
    identity_score,
    mota_score,
)
from soccertrack.tracking_model import (
    KalmanTracker,
    MultiObjectTracker,
)
from soccertrack.tracking_model.matching import MotionVisualMatchingFunction
from soccertrack.utils import get_git_root
from soccertrack.types import _pathlike
from soccertrack.logger import logger, set_log_level, inspect

import argparse
from omegaconf import OmegaConf
import re


def is_camel_case(s: str) -> bool:
    return (
        s != s.lower()
        and s != s.upper()
        and "_" not in s
        and bool(re.match(r"[A-Za-z0-9]+", s))
    )


def create_instance(class_name: str, *args, **kwargs):
    module = __import__(__name__)
    class_ = getattr(module, class_name)
    instance = class_(*args, **kwargs)
    return instance


def create_matching_fn(matching_fn_conf: dict):
    module_name = "soccertrack.tracking_model.matching"
    matching_fn_module = importlib.import_module(module_name)

    matching_fn_name = matching_fn_conf.pop("name")
    matching_fn_class = getattr(matching_fn_module, matching_fn_name)

    # Initialize the kwargs for the matching function
    matching_fn_kwargs = {}
    for key, value in matching_fn_conf.items():
        if is_camel_case(str(value)):
            class_name = value
            class_ = getattr(matching_fn_module, class_name)
            instance = class_()
            matching_fn_kwargs[key] = instance
        else:
            matching_fn_kwargs[key] = value

    # Instantiate the matching function
    matching_fn = matching_fn_class(**matching_fn_kwargs)

    return matching_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="A script to run object detection, tracking, and image classification."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML configuration file (default: None).",
    )

    parser.add_argument(
        "--path_to_mp4",
        help="Path to the video MP4 file",
        default=None,
    )
    parser.add_argument(
        "--path_to_csv",
        help="Path to the annotations CSV files",
        default=None,
    )
    parser.add_argument(
        "--det_model_name",
        default="yolov5",
        help="Name of the object detection model to use (default: yolov5).",
    )
    parser.add_argument(
        "--det_model_repo",
        default="external/yolov5",
        help="Path to the object detection model repository (default: external/yolov5).",
    )
    parser.add_argument(
        "--det_model_ckpt",
        default="models/yolov5/yolov5x_last.pt",
        help="Path to the object detection model checkpoint (default: models/yolov5/yolov5x_last.pt).",
    )
    parser.add_argument(
        "--img_model_name",
        default="resnet18",
        help="Name of the image classification model to use (default: resnet18).",
    )
    parser.add_argument(
        "--img_model_ckpt",
        default="models/resnet18/last.ckpt",
        help="Path to the image classification model checkpoint (default: models/resnet18/last.ckpt).",
    )
    parser.add_argument(
        "--sot_tracker",
        default="KalmanTracker",
        help="Name of the single object tracker to use (default: KalmanTracker).",
    )
    parser.add_argument(
        "--sot_tracker_kwargs",
        default="{}",
        help="JSON string with keyword arguments for the single object tracker (default: {}).",
    )
    parser.add_argument(
        "--frame_skip",
        default=1,
        type=int,
        help="Number of frames to skip between detections (default: 1).",
    )
    parser.add_argument(
        "--num_frames",
        default=100,
        type=int,
        help="Number of frames to process (default: 100).",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--save_path",
        default="output/deepsort",
        help="Path to save the output results (default: output/deepsort).",
    )
    parser.add_argument(
        "--save_results",
        default=False,
        action="store_true",
        help="Whether to save the tracking results (default: False).",
    )
    parser.add_argument(
        "--save_video",
        default=False,
        action="store_true",
        help="Whether to save the processed video (default: False).",
    )
    parser.add_argument(
        "--save_video_kwargs",
        default="{}",
        help="JSON string with keyword arguments for saving video (default: {}).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run the model on, either 'cuda' or 'cpu' (default: cuda).",
    )
    parser.add_argument(
        "--matching_fn",
        default=None,
        type=str,
        help="JSON string with matching function configuration (default: None).",
    )

    args = parser.parse_args()

    # Convert argparse.Namespace to a dictionary
    args_dict = vars(args)

    # Load the configuration from the YAML file if provided
    if args.config is not None:
        file_config = OmegaConf.load(args.config)
        config = OmegaConf.merge(OmegaConf.create(args_dict), file_config)
    else:
        config = OmegaConf.create(args_dict)

    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict.pop("config")
    return config_dict


def run(
    path_to_csv: _pathlike = None,
    path_to_mp4: _pathlike = None,
    det_model_name: str = "yolov5",
    det_model_repo: _pathlike = "external/yolov5",
    det_model_ckpt: _pathlike = "models/yolov5/yolov5x_last.pt",
    det_model_conf: dict = {},
    img_model_name: str = "resnet18",
    img_model_ckpt: str = "",
    sot_tracker: str = "KalmanTracker",
    sot_tracker_kwargs: str = "{}",
    matching_fn: dict = None,
    frame_skip: int = 1,
    num_frames: int = 100,
    save_path: _pathlike = "output/deepsort",
    save_results: bool = False,
    save_video: bool = False,
    save_video_kwargs: Optional[dict] = None,
    device: str = "cuda",
):
    root = get_git_root()
    print(img_model_name)

    # Define the object detection model
    det_model = detection_model.load(det_model_name, det_model_repo, det_model_ckpt, det_model_conf)

    # Define the image embedding model
    if img_model_name is None:
        # If no image model is provided, we will not use image embeddings (ex. SORT)
        image_model = None
    else:
        image_model = TorchReIDModel(
            model_name=img_model_name, model_path=img_model_ckpt, device=device
        )

    # Define the KalmanTracker
    """for spec = {'order_pos': 1, 'dim_pos': 2, 'order_size': 0, 'dim_size': 1}
    we expect the following setup:
    state x, x', y, y', w, h
    where x and y are centers of boxes, w and h are width and height"""
    sot_tracker = KalmanTracker
    sot_tracker_kwargs = {
        "model_kwargs": {
            "dt": 1 / 25,
            "order_pos": 1,
            "dim_pos": 2,
            "order_size": 0,
            "dim_size": 2,
            "q_var_pos": 70.0,
            "r_var_pos": 1.0,
            "q_var_size": 20.0,
            "r_var_size": 1.0,
            "p_cov_p0": 0.0012,
        }
    }

    # Define the Matching Function
    matching_fn = create_matching_fn(matching_fn)

    # Define the MultiObjectTracker
    tracker = MultiObjectTracker(
        detection_model=det_model,
        image_model=image_model,
        tracker=sot_tracker,
        tracker_kwargs=sot_tracker_kwargs,
        matching_fn=matching_fn,
    )

    # Define the camera
    cam = Camera(path_to_mp4)  # Camera object will be used to load frames

    # Run the tracker
    tracker.track(cam, num_frames=num_frames)

    # Evaluate the tracker
    save_path = root / Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    pred = tracker.to_bbdf()

    if path_to_csv is not None:
        gt = soccertrack.load_df(path_to_csv)  # We will use this as ground truth

        mota = mota_score(bboxes_gt=gt, bboxes_track=pred)
        hota = hota_score(bboxes_gt=gt, bboxes_track=pred)
        identity = identity_score(bboxes_gt=gt, bboxes_track=pred)

        print(mota)
        print(hota)
        print(identity)

    if save_results:
        pred.to_csv(save_path / "predictions.csv")

    if save_video:
        pred.visualize_frames(
            cam.video_path, save_path / "video.mp4", **save_video_kwargs
        )


if __name__ == "__main__":
    config_dict = parse_args()

    set_log_level(config_dict.pop("log_level"))

    # Print key-value pairs without brackets
    print("Configuration:")
    for key, value in config_dict.items():
        print(f"{key}: {value}")

    run(**config_dict)
