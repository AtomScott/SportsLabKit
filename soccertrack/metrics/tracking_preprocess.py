from __future__ import annotations

from itertools import chain
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from soccertrack import BBoxDataFrame
from soccertrack.metrics.object_detection import convert_to_x1y1x2y2, iou_score


def to_mot_eval_format(
    gt_bbdf: BBoxDataFrame,
    pred_bbdf: BBoxDataFrame,
) -> dict[str, Any]:
    """Converts tracking and ground truth data to the format(dictionary)
    required by the MOT metrics.

    Args:
        gt_bbdf (BBoxDataFrame): Bbox Dataframe for ground truth tracking data.
        pred_bbdf (BBoxDataFrame): Bbox Dataframe for predicted tracking data.

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
    if gt_bbdf.size == 0 and pred_bbdf.size == 0:
        data = {}
        data["tracker_ids"] = []
        data["gt_ids"] = 0
        data["tracker_dets"] = []
        data["gt_dets"] = 0
        data["similarity_scores"] = []
        data["num_tracker_dets"] = 0
        data["num_gt_dets"] = 0
        data["num_tracker_ids"] = 0
        data["num_gt_ids"] = 0
        data["num_timesteps"] = 0
        return data

    min_frame = min(
        gt_bbdf.first_valid_index() or pred_bbdf.first_valid_index(),
        pred_bbdf.first_valid_index() or gt_bbdf.first_valid_index(),
    )
    max_frame = max(
        gt_bbdf.last_valid_index() or pred_bbdf.last_valid_index(),
        pred_bbdf.last_valid_index() or gt_bbdf.last_valid_index(),
    )

    pred_bbdf = pred_bbdf.reindex(range(min_frame, max_frame + 1))
    gt_bbdf = gt_bbdf.reindex(range(min_frame, max_frame + 1))

    assert pred_bbdf.index.equals(
        gt_bbdf.index
    ), f"Index mismatch: {pred_bbdf.index} != {gt_bbdf.index}"

    gt_ids, gt_dets = gt_bbdf.preprocess_for_mot_eval()
    pred_ids, pred_dets = pred_bbdf.preprocess_for_mot_eval()
    
    print(pred_bbdf)
    print(pred_ids)

    num_tracker_dets = sum(len(pred_dets[i]) for i in range(len(pred_dets)))
    num_gt_dets = sum(len(gt_dets[i]) for i in range(len(gt_dets)))

    unique_tracker_ids = np.unique(list(chain.from_iterable(pred_ids)))
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
        data["tracker_ids"] = pred_ids
        data["gt_ids"] = []
        data["tracker_dets"] = pred_dets
        data["gt_dets"] = []
        data["similarity_scores"] = []
        data["num_tracker_dets"] = num_tracker_dets
        data["num_gt_dets"] = 0
        data["num_tracker_ids"] = len(unique_tracker_ids)
        data["num_gt_ids"] = 0
        data["num_timesteps"] = 0
        return data

    tracker_dets_xyxy = [
        [convert_to_x1y1x2y2(bbox) for bbox in frame_dets] for frame_dets in pred_dets
    ]
    gt_dets_xyxy = [
        [convert_to_x1y1x2y2(bbox) for bbox in frame_dets] for frame_dets in gt_dets
    ]

    sim_score_list = []
    for i in range(len(gt_ids)):

        if len(gt_ids[i]) == 0 or len(pred_ids[i]) == 0:
            sim_score_list.append(np.array([]))
        else:
            sim_score = cdist(gt_dets_xyxy[i], tracker_dets_xyxy[i], iou_score)
            sim_score_list.append(sim_score)

    data = {}
    data["tracker_ids"] = pred_ids
    data["gt_ids"] = gt_ids
    data["tracker_dets"] = pred_dets
    data["gt_dets"] = gt_dets
    data["similarity_scores"] = sim_score_list
    data["num_tracker_dets"] = num_tracker_dets
    data["num_gt_dets"] = num_gt_dets
    data["num_tracker_ids"] = len(unique_tracker_ids)
    data["num_gt_ids"] = len(unique_gt_ids)
    data["num_timesteps"] = len(gt_dets)
    return data
