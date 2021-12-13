"""Tools to retrieve bounding boxes from a given object detector."""

import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from more_itertools import bucket, chunked
from numpy.typing import NDArray
from PIL import Image
from podm.bounding_box import BoundingBox
from podm.utils.enumerators import BBFormat, BBType, CoordinatesType
from tqdm.auto import tqdm

from soccertrack.utils.camera import Camera
from soccertrack.utils.detection import CandidateDetection


def detect_objects(
    cameras: List[Camera],
    model_name: str,
    batch_size: int,
    size: Optional[int],
    filter_range: bool = True,
) -> Tuple[Dict[int, List[CandidateDetection]], Dict[int, List[CandidateDetection]]]:
    """Detect object from a list of camera objects.

    Args:
        cameras (List[Camera]): List of cameras
        model_name (str): Name of the model to use
        batch_size (int): Batch size
        size (Optional[int]): list of lists of detections sorted by frame number
        filter_range (bool): Filter detections by range specified in the camera object

    Returns:
        List[List[CandidateDetection]]: List of lists of detections sorted by frame number
    """
    ball_candidate_detections = []
    person_candidate_detections = []
    model, input_func, output_func = get_detection_model(model_name)
    for camera in cameras:
        bounding_boxes_list = []
        movie_iterator = camera.movie_iterator(True)

        for batch in chunked(tqdm(movie_iterator), batch_size):
            batch_input = input_func(batch)
            results = output_func(model(batch_input, size=size))
            bounding_boxes_list.extend(results)

        for frame_idx, bounding_boxes in enumerate(bounding_boxes_list):
            for bbox in bounding_boxes:
                candidate_detection = CandidateDetection(
                    from_bbox=bbox, camera=camera, frame_idx=frame_idx
                )

                if (not candidate_detection.in_range) and filter_range:
                    continue
                if candidate_detection._class_id == 32:
                    ball_candidate_detections.append(candidate_detection)
                elif candidate_detection._class_id == 0:
                    person_candidate_detections.append(candidate_detection)
                else:
                    warnings.warn(f"Unknown class id {candidate_detection._class_id}")

    person_candidate_detections_dict = sort_candidates(person_candidate_detections)
    ball_candidate_detections_dict = sort_candidates(ball_candidate_detections)

    return person_candidate_detections_dict, ball_candidate_detections_dict


def sort_candidates(
    candidate_detections: List[CandidateDetection],
) -> Dict[int, List[CandidateDetection]]:
    """Sort candidate detections by frame number.

    Args:
        candidate_detections (List[CandidateDetection]): List of candidate detections

    Returns:
        Dict[int, List[CandidateDetection]]: Dictionary of lists of candidate detections sorted by frame number
    """
    candidate_detections.sort(
        key=lambda candidate_detection: candidate_detection.frame_idx
    )

    buckets = bucket(
        candidate_detections,
        key=lambda candidate_detection: candidate_detection.frame_idx,
    )

    dict_of_detections = {frame_idx: list(buckets[frame_idx]) for frame_idx in buckets}
    return dict_of_detections


# TODO: decorator to verify model output
# @verify_model_input
def yolov5_input_func(batch_input: Iterable[NDArray[np.uint8]]) -> Iterable[Any]:
    """Convert input to yolo v5 model input.

    Args:
        batch_input (Iterable): List of images

    Returns:
        List[Any]: A list of PIL images or RGB cv2 images
    """
    assert any(
        [
            isinstance(batch_input, list),
            isinstance(batch_input, tuple),
        ]
    ), type(batch_input)

    assert any(
        [
            *[isinstance(img, Image.Image) for img in batch_input],
            *[isinstance(img, np.ndarray) for img in batch_input],
        ]
    ), [type(img) for img in batch_input]
    return batch_input


# @verify_model_output
def yolov5_output_func(batch_output: Any) -> List[List[BoundingBox]]:
    """Convert output from yolo v5 model to a list of lists of bounding box objects.

    Args:
        batch_output (Any): Output from yolo v5 model

    Returns:
        List[List[BoundingBox]]: List of lists of bounding box objects
    """
    bounding_boxes_list = []
    for frame_idx, dets in enumerate(batch_output.pandas().xyxy):
        bounding_boxes = []
        for _, det in dets.iterrows():

            bbox = BoundingBox(
                image_name=batch_output.files[frame_idx],
                class_id=det["class"],
                coordinates=(det.xmin, det.ymin, det.xmax, det.ymax),
                type_coordinates=CoordinatesType.ABSOLUTE,
                img_size=batch_output.imgs[frame_idx].shape[:2],
                bb_type=BBType.DETECTED,
                confidence=det.confidence,
                format=BBFormat.XYX2Y2,
            )

            bounding_boxes.append(bbox)
        bounding_boxes_list.append(bounding_boxes)
    return bounding_boxes_list


def get_detection_model(model_name: str) -> Any:
    """Wrap models to return bounding box objects.

    Args:
        model_name (str): Name of the model to use

    Raises:
        ValueError: If model name is not supported

    Returns:
        Any: Model, input function and output function
    """
    #

    if model_name == "yolov5s":
        input_func = yolov5_input_func
        output_func = yolov5_output_func
        return (
            torch.hub.load("ultralytics/yolov5", "yolov5s"),
            input_func,
            output_func,
        )  #  P6 model
    if model_name == "yolov5m":
        input_func = yolov5_input_func
        output_func = yolov5_output_func
        return (
            torch.hub.load("ultralytics/yolov5", "yolov5m"),
            input_func,
            output_func,
        )  #  P6 model
    if model_name == "yolov5x":
        input_func = yolov5_input_func
        output_func = yolov5_output_func
        return (
            torch.hub.load("ultralytics/yolov5", "yolov5x"),
            input_func,
            output_func,
        )  #  P6 model

    raise ValueError(f"Unknown model name: {model_name}")
