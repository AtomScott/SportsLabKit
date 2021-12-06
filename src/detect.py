import warnings
from typing import Any, Dict, List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from more_itertools import bucket, chunked
from mplsoccer import Pitch, VerticalPitch
from PIL import Image
from podm.bounding_box import BoundingBox
from podm.utils.enumerators import BBFormat, BBType, CoordinatesType
from tqdm.auto import tqdm

from src.utils import MovieIterator
from src.utils.camera import Camera
from src.utils.detection import CandidateDetection


def detect_objects(
    cameras: List[Camera],
    model_name: str,
    batch_size: int,
    size: Optional[int],
    filter_range: bool = True,
) -> Union[Dict[int, List[CandidateDetection]], Dict[int, List[CandidateDetection]]]:
    """[summary]

    Args:
        cameras (List[Camera]): [description]
        model_name (str): [description]
        batch_size (int): [description]
        size (Optional[int]): list of lists of detections sorted by frame number

    Returns:
        List[List[CandidateDetection]]: [description]
    """
    ball_candidate_detections = []
    person_candidate_detections = []
    model, input_func, output_func = get_detection_model(model_name)
    for camera in cameras:
        bounding_boxes_list = []
        frame_idxs = []
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

    ball_candidate_detections = sort_candidates(ball_candidate_detections)
    person_candidate_detections = sort_candidates(person_candidate_detections)

    return person_candidate_detections, ball_candidate_detections


def sort_candidates(
    candidate_detections: List[CandidateDetection],
) -> Dict[int, List[CandidateDetection]]:

    # sort detections by frame number
    # return list of lists of detections

    candidate_detections.sort(
        key=lambda candidate_detection: candidate_detection.frame_idx
    )

    buckets = bucket(
        candidate_detections,
        key=lambda candidate_detection: candidate_detection.frame_idx,
    )

    dict_of_detections = {frame_idx: list(buckets[frame_idx]) for frame_idx in buckets}
    return dict_of_detections
    # return buckets
    # df_ball = pd.DataFrame()
    # df_person = pd.DataFrame()

    # for i, batch in enumerate(chunked(tqdm(movie_iterator), batch_size)):
    #     if fast:
    #         results = model(batch, size=500, augment=False)
    #     else:
    #         results = model(batch, size=movie_iterator.img_width, augment=True)

    #     for j, row in enumerate(results.pandas().xyxy):
    #         row["frame"] = i * batch_size + j
    #         df_ball = pd.concat([df_ball, row[row["name"] == "sports ball"]])
    #         df_person = pd.concat([df_person, row[row["name"] == "person"]])
    # return df_person, df_ball


# @verify_model_input
def yolov5_input_func(batch_input) -> List[Any]:
    # model input is a list of PIL images or RGB cv2 images
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
def yolov5_output_func(batch_output):
    bounding_boxes_list = []
    for frame_idx, dets in enumerate(batch_output.pandas().xywhn):
        bounding_boxes = []
        for det_idx, det in dets.iterrows():

            bbox = BoundingBox(
                image_name=batch_output.files[frame_idx],
                class_id=det["class"],
                coordinates=(det.xcenter, det.ycenter, det.width, det.height),
                type_coordinates=CoordinatesType.RELATIVE,
                img_size=batch_output.imgs[frame_idx].shape[:2],
                bb_type=BBType.DETECTED,
                confidence=det.confidence,
                format=BBFormat.YOLO,
            )

            bounding_boxes.append(bbox)
        bounding_boxes_list.append(bounding_boxes)
    return bounding_boxes_list


def get_detection_model(model_name: str) -> Any:
    # wrap models so that they return bounding box objects

    if model_name == "yolov5s":
        input_func = yolov5_input_func
        output_func = yolov5_output_func
        return (
            torch.hub.load("ultralytics/yolov5", "yolov5s"),
            input_func,
            output_func,
        )  #  P6 model
    elif model_name == "yolov5x":
        input_func = yolov5_input_func
        output_func = yolov5_output_func
        return (
            torch.hub.load("ultralytics/yolov5", "yolov5x"),
            input_func,
            output_func,
        )  #  P6 model
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# TODO: add visualization of detections
# def visualize_results(videopath, df_ball, df_person, outpath):
#     imgs = []
#     movie_iterator = MovieIterator(videopath)
#     for frame_idx, img in enumerate(tqdm(movie_iterator)):
#         frame_df_ball = df_ball[df_ball["frame"] == frame_idx][
#             ["xmin", "ymin", "xmax", "ymax", "confidence", "name"]
#         ]
#         frame_df_person = df_person[df_person["frame"] == frame_idx][
#             ["xmin", "ymin", "xmax", "ymax", "confidence", "name"]
#         ]

#         for row_idx, row in frame_df_person.iterrows():
#             color = (0, 0, 255)
#             confidence = row.pop("confidence")
#             label = row.pop("name")
#             xmin, ymin, xmax, ymax = pd.to_numeric(row).values.round().astype(int)
#             img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color)
#             cv2.putText(
#                 img,
#                 f"{label} {confidence:.3f}",
#                 (xmin, ymin - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.9,
#                 color,
#                 2,
#             )

#         for row_idx, row in frame_df_ball.iterrows():
#             color = (0, 255, 0)
#             confidence = row.pop("confidence")
#             label = row.pop("name")
#             xmin, ymin, xmax, ymax = pd.to_numeric(row).values.round().astype(int)
#             img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color)
#             cv2.putText(
#                 img,
#                 f"{label} {confidence:.3f}",
#                 (xmin, ymin - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.9,
#                 color,
#                 2,
#             )

#         imgs.append(img)
#     make_video(imgs, movie_iterator.video_fps, outpath)
