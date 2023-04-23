import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from omegaconf import OmegaConf

import soccertrack
from soccertrack import Camera, detection_model
from soccertrack.logger import inspect, logger, set_log_level, tqdm
from soccertrack.metrics.object_detection import map_score
from soccertrack.types import _pathlike
from soccertrack.utils import get_git_root, make_video
from soccertrack.utils.draw import draw_bounding_boxes


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


def draw_func(results, cam, **kwargs):
    for result in results:
        im = result.show()
        yield np.array(im)


def run(
    path_to_csv: _pathlike = None,
    path_to_mp4: _pathlike = None,
    det_model_name: str = "yolov5",
    det_model_repo: _pathlike = "external/yolov5",
    det_model_ckpt: _pathlike = "models/yolov5/yolov5x_last.pt",
    save_path: _pathlike = "output/deepsort",
    save_results: bool = False,
    save_video: bool = False,
    save_video_kwargs: Optional[dict] = None,
    device: str = "cuda",
    batch_size: int = 1,
):
    root = get_git_root()

    # Define the object detection model
    det_model = detection_model.load(det_model_name, det_model_repo, det_model_ckpt)

    # Define the camera
    cam = Camera(path_to_mp4)  # Camera object will be used to load frames

    # Perform detection
    results = []
    for frames in tqdm(cam[:4]):
        det_result = det_model(frames, augment=True, size=6500)
        results.append(det_result)

    # Evaluate the detections
    save_path = root / Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    if path_to_csv is not None:
        raise NotImplementedError("Evaluation is not implemented yet.")

    if save_results:
        raise NotImplementedError("Saving results is not implemented yet.")

    if save_video:
        make_video(
            draw_func(results, cam, colors="red", width=5),
            outpath=save_path / "video.mp4",
            input_framerate=30,
            **save_video_kwargs,
        )


if __name__ == "__main__":
    config_dict = parse_args()

    set_log_level(config_dict.pop("log_level"))

    # Print key-value pairs without brackets
    print("Configuration:")
    for key, value in config_dict.items():
        print(f"{key}: {value}")

    run(**config_dict)
