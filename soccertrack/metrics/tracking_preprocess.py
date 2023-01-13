from __future__ import annotations

from itertools import chain
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from soccertrack.metrics import iou_score
from soccertrack.metrics.object_detection import convert_to_x1y1x2y2


def to_mot_eval_format(
    tracker_ids: list[list[int]],
    tracker_dets: list[list[np.ndarray]],
    gt_ids: list[list[int]],
    gt_dets: list[list[np.ndarray]],
) -> dict[str, Any]:
    """Converts tracking and ground truth data to the format(dictionary) required by the MOT metrics.

    Args:
        bboxes_gt (BBoxDataFrame): Bbox Dataframe for ground truth in 1 sequence
        tracker_ids (list[list[int]]): List of lists of tracker ids for each timestep
        tracker_dets (list[list[np.ndarray]]): List of lists of tracker detections for each timestep
        tracker_dets_xyxy (list[list[np.ndarray]]): List of lists of tracker detections in xyxy format for each timestep
        gt_ids (list[list[int]]): List of lists of ground truth ids for each timestep
        gt_dets (list[list[np.ndarray]]): List of lists of ground truth detections for each timestep
        gt_dets_xyxy (list[list[np.ndarray]]): List of lists of ground truth detections in xyxy format for each timestep

    Returns:
        dict[str, Any]: Dictionary containing the data required by the MOT metrics

    Note:
    data is a dict containing all of the information that metrics need to perform evaluation.
    It contains the following fields:
        [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
        [gt_ids, tracker_ids]: list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, tracker_dets]: list (for each timestep) of lists of detection masks.
        [similarity_scores]: list (for each timestep) of 2D NDArrays.

    reference : https://github.com/JonathonLuiten/TrackEval/blob/ec237ec3ef654548fdc1fa1e100a45b31a6d4499/trackeval/datasets/mots_challenge.py
    """

    num_tracker_dets = sum([len(tracker_dets[i]) for i in range(len(tracker_dets))])
    num_gt_dets = sum([len(gt_dets[i]) for i in range(len(gt_dets))])

    unique_tracker_ids = np.unique(list(chain.from_iterable(tracker_ids)))
    unique_gt_ids = np.unique(list(chain.from_iterable(gt_ids)))

    if num_tracker_dets == 0:
        data = {}
        data["tracker_ids"] = []
        data["gt_ids"] = gt_ids
        data["tracker_dets"] = []
        data["gt_dets"] = gt_dets
        data["similarity_scores"] = []
        data["num_tracker_dets"] = 0
        data["num_gt_dets"] = num_gt_dets
        data["num_tracker_ids"] = 0
        data["num_gt_ids"] = len(unique_gt_ids)
        data["num_timesteps"] = len(gt_dets)
        return data

    if num_gt_dets == 0:
        data = {}
        data["tracker_ids"] = tracker_ids
        data["gt_ids"] = []
        data["tracker_dets"] = tracker_dets
        data["gt_dets"] = []
        data["similarity_scores"] = []
        data["num_tracker_dets"] = num_tracker_dets
        data["num_gt_dets"] = 0
        data["num_tracker_ids"] = len(unique_tracker_ids)
        data["num_gt_ids"] = 0
        data["num_timesteps"] = 0
        return data

    tracker_dets_xyxy = [
        [convert_to_x1y1x2y2(bbox) for bbox in frame_dets]
        for frame_dets in tracker_dets
    ]
    gt_dets_xyxy = [
        [convert_to_x1y1x2y2(bbox) for bbox in frame_dets] for frame_dets in gt_dets
    ]

    sim_score_list = []
    for i in range(len(gt_ids)):
        sim_score = cdist(gt_dets_xyxy[i], tracker_dets_xyxy[i], iou_score)
        sim_score_list.append(sim_score)

    data = {}
    data["tracker_ids"] = tracker_ids
    data["gt_ids"] = gt_ids
    data["tracker_dets"] = tracker_dets
    data["gt_dets"] = gt_dets
    data["similarity_scores"] = sim_score_list
    data["num_tracker_dets"] = num_tracker_dets
    data["num_gt_dets"] = num_gt_dets
    data["num_tracker_ids"] = len(unique_tracker_ids)
    data["num_gt_ids"] = len(unique_gt_ids)
    data["num_timesteps"] = len(gt_dets)
    return data
