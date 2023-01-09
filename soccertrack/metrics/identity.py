from __future__ import annotations
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from soccertrack import BBoxDataFrame
from .tracking_preprocess import to_mot_eval_format

def identity_score(
    bboxes_track: BBoxDataFrame, bboxes_gt: BBoxDataFrame
) -> dict[str, Any]:
    """Calculates ID metrics for one sequence.

    Args:
        bboxes_track (BBoxDataFrame): Bbox Dataframe for tracking in 1 sequence
        bboxes_gt (BBoxDataFrame): Bbox Dataframe for ground truth in 1 sequence

    Returns:
        dict[str, Any]: ID metrics
    """
    
    tracker_ids, tracker_dets = bboxes_track.preprocess_for_mot_eval()
    gt_ids, gt_dets = bboxes_gt.preprocess_for_mot_eval()
    
    data = to_mot_eval_format(tracker_ids, tracker_dets, gt_ids, gt_dets)

    integer_fields = ["IDTP", "IDFN", "IDFP"]
    float_fields = ["IDF1", "IDR", "IDP"]
    fields = float_fields + integer_fields
    threshold = 0.5

    # Initialise results
    res = {}
    for field in fields:
        res[field] = 0

    # Return result quickly if tracker or gt sequence is empty
    if data["num_tracker_dets"] == 0:
        res["IDFN"] = data["num_gt_dets"]
        # Calculate final scores
        id_final_scores(res)
        return res
    
    if data["num_gt_dets"] == 0:
        res["IDFP"] = data["num_tracker_dets"]
        # Calculate final scores
        id_final_scores(res)
        return res

    # Variables counting global association
    potential_matches_count = np.zeros((data["num_gt_ids"], data["num_tracker_ids"]))
    gt_id_count = np.zeros(data["num_gt_ids"])
    tracker_id_count = np.zeros(data["num_tracker_ids"])

    # First loop through each timestep and accumulate global track information.
    for t, (gt_ids_t, tracker_ids_t) in enumerate(
        zip(data["gt_ids"], data["tracker_ids"])
    ):
        # Count the potential matches between ids in each timestep
        matches_mask = np.greater_equal(data["similarity_scores"][t], threshold)
        match_idx_gt, match_idx_tracker = np.nonzero(matches_mask)
        potential_matches_count[
            gt_ids_t[match_idx_gt], tracker_ids_t[match_idx_tracker]
        ] += 1

        # Calculate the total number of dets for each gt_id and tracker_id.
        gt_id_count[gt_ids_t] += 1
        tracker_id_count[tracker_ids_t] += 1

    # Calculate optimal assignment cost matrix for ID metrics
    num_gt_ids = data["num_gt_ids"]
    num_tracker_ids = data["num_tracker_ids"]
    fp_mat = np.zeros((num_gt_ids + num_tracker_ids, num_gt_ids + num_tracker_ids))
    fn_mat = np.zeros((num_gt_ids + num_tracker_ids, num_gt_ids + num_tracker_ids))
    fp_mat[num_gt_ids:, :num_tracker_ids] = 1e10
    fn_mat[:num_gt_ids, num_tracker_ids:] = 1e10
    for gt_id in range(num_gt_ids):
        fn_mat[gt_id, :num_tracker_ids] = gt_id_count[gt_id]
        fn_mat[gt_id, num_tracker_ids + gt_id] = gt_id_count[gt_id]
    for tracker_id in range(num_tracker_ids):
        fp_mat[:num_gt_ids, tracker_id] = tracker_id_count[tracker_id]
        fp_mat[tracker_id + num_gt_ids, tracker_id] = tracker_id_count[tracker_id]
    fn_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count
    fp_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count

    # Hungarian algorithm
    match_rows, match_cols = linear_sum_assignment(fn_mat + fp_mat)

    # Accumulate basic statistics
    res["IDFN"] = fn_mat[match_rows, match_cols].sum().astype(int)
    res["IDFP"] = fp_mat[match_rows, match_cols].sum().astype(int)
    res["IDTP"] = (gt_id_count.sum() - res["IDFN"]).astype(int)
    # Calculate final ID scores
    id_final_scores(res)
    return res

def id_final_scores(res):
    res["IDR"] = res["IDTP"] / np.maximum(1.0, res["IDTP"] + res["IDFN"])
    res["IDP"] = res["IDTP"] / np.maximum(1.0, res["IDTP"] + res["IDFP"])
    res["IDF1"] = res["IDTP"] / np.maximum(
        1.0, res["IDTP"] + 0.5 * res["IDFP"] + 0.5 * res["IDFN"]
    )  