from __future__ import annotations
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from soccertrack import BBoxDataFrame
from .tracking_preprocess import to_mot_eval_format

def hota_score(bboxes_track: BBoxDataFrame, bboxes_gt: BBoxDataFrame) -> dict[str, Any]:
    """Calculates the HOTA metrics for one sequence.

    Args:
        bboxes_track (BBoxDataFrame): Bbox Dataframe for tracking in 1 sequence
        bboxes_gt (BBoxDataFrame): Bbox Dataframe for ground truth in 1 sequence

    Returns:
        dict[str, Any]: HOTA metrics
    """
    
    tracker_ids, tracker_dets = bboxes_track.preprocess_for_mot_eval()
    gt_ids, gt_dets = bboxes_gt.preprocess_for_mot_eval()
    data = to_mot_eval_format(tracker_ids, tracker_dets, gt_ids, gt_dets)

    array_labels = np.arange(0.05, 0.99, 0.05)
    integer_array_fields = ["HOTA_TP", "HOTA_FN", "HOTA_FP"]
    float_array_fields = [
        "HOTA",
        "DetA",
        "AssA",
        "DetRe",
        "DetPr",
        "AssRe",
        "AssPr",
        "LocA",
        "RHOTA",
    ]
    float_fields = ["HOTA(0)", "LocA(0)", "HOTALocA(0)"]

    # Initialise results
    res = {}
    for field in float_array_fields + integer_array_fields:
        res[field] = np.zeros((len(array_labels)), dtype=np.float)
    for field in float_fields:
        res[field] = 0

    # Return result quickly if tracker or gt sequence is empty
    if data["num_tracker_dets"] == 0:
        res["HOTA_FN"] = data["num_gt_dets"] * np.ones(
            (len(array_labels)), dtype=np.float
        )
        res["LocA"] = np.ones((len(array_labels)), dtype=np.float)
        res["LocA(0)"] = 1.0
        # Calculate final scores
        hota_final_scores(res)
        return res

    if data["num_gt_dets"] == 0:
        res["HOTA_FP"] = data["num_tracker_dets"] * np.ones(
            (len(array_labels)), dtype=np.float
        )
        res["LocA"] = np.ones((len(array_labels)), dtype=np.float)
        res["LocA(0)"] = 1.0
        # Calculate final scores
        hota_final_scores(res)
        return res

    # Variables counting global association
    potential_matches_count = np.zeros((data["num_gt_ids"], data["num_tracker_ids"]))
    gt_id_count = np.zeros((data["num_gt_ids"], 1))
    tracker_id_count = np.zeros((1, data["num_tracker_ids"]))

    # First loop through each timestep and accumulate global track information.
    for t, (gt_ids_t, tracker_ids_t) in enumerate(
        zip(data["gt_ids"], data["tracker_ids"])
    ):
        # Count the potential matches between ids in each timestep
        # These are normalised, weighted by the match similarity.
        similarity = data["similarity_scores"][t]
        sim_iou_denom = (
            similarity.sum(0)[np.newaxis, :]
            + similarity.sum(1)[:, np.newaxis]
            - similarity
        )
        sim_iou = np.zeros_like(similarity)
        sim_iou_mask = sim_iou_denom > 0 + np.finfo("float").eps
        sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
        potential_matches_count[
            gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]
        ] += sim_iou
        # Calculate the total number of dets for each gt_id and tracker_id.
        gt_id_count[gt_ids_t] += 1
        tracker_id_count[0, tracker_ids_t] += 1

    # Calculate overall jaccard alignment score (before unique matching) between IDs
    global_alignment_score = potential_matches_count / (
        gt_id_count + tracker_id_count - potential_matches_count
    )
    matches_counts = [np.zeros_like(potential_matches_count) for _ in array_labels]

    # Calculate scores for each timestep
    for t, (gt_ids_t, tracker_ids_t) in enumerate(
        zip(data["gt_ids"], data["tracker_ids"])
    ):
        # Deal with the case that there are no gt_det/tracker_det in a timestep.
        if len(gt_ids_t) == 0:
            for a, alpha in enumerate(array_labels):
                res["HOTA_FP"][a] += len(tracker_ids_t)
            continue
        if len(tracker_ids_t) == 0:
            for a, alpha in enumerate(array_labels):
                res["HOTA_FN"][a] += len(gt_ids_t)
            continue

        # Get matching scores between pairs of dets for optimizing HOTA
        similarity = data["similarity_scores"][t]
        score_mat = (
            global_alignment_score[
                gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]
            ]
            * similarity
        )

        # Hungarian algorithm to find best matches
        match_rows, match_cols = linear_sum_assignment(-score_mat)

        # Calculate and accumulate basic statistics
        for a, alpha in enumerate(array_labels):
            actually_matched_mask = (
                similarity[match_rows, match_cols] >= alpha - np.finfo("float").eps
            )
            alpha_match_rows = match_rows[actually_matched_mask]
            alpha_match_cols = match_cols[actually_matched_mask]
            num_matches = len(alpha_match_rows)
            res["HOTA_TP"][a] += num_matches
            res["HOTA_FN"][a] += len(gt_ids_t) - num_matches
            res["HOTA_FP"][a] += len(tracker_ids_t) - num_matches
            if num_matches > 0:
                res["LocA"][a] += sum(similarity[alpha_match_rows, alpha_match_cols])
                matches_counts[a][
                    gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols]
                ] += 1

    # Calculate association scores (AssA, AssRe, AssPr) for the alpha value.
    # First calculate scores per gt_id/tracker_id combo and then average over the number of detections.
    for a, alpha in enumerate(array_labels):
        matches_count = matches_counts[a]
        ass_a = matches_count / np.maximum(
            1, gt_id_count + tracker_id_count - matches_count
        )
        res["AssA"][a] = np.sum(matches_count * ass_a) / np.maximum(
            1, res["HOTA_TP"][a]
        )
        ass_re = matches_count / np.maximum(1, gt_id_count)
        res["AssRe"][a] = np.sum(matches_count * ass_re) / np.maximum(
            1, res["HOTA_TP"][a]
        )
        ass_pr = matches_count / np.maximum(1, tracker_id_count)
        res["AssPr"][a] = np.sum(matches_count * ass_pr) / np.maximum(
            1, res["HOTA_TP"][a]
        )

    # Calculate scores for each alpha value
    res["LocA"] = np.maximum(1e-10, res["LocA"]) / np.maximum(1e-10, res["HOTA_TP"])

    res["DetRe"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"])
    res["DetPr"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FP"])
    res["DetA"] = res["HOTA_TP"] / np.maximum(
        1, res["HOTA_TP"] + res["HOTA_FN"] + res["HOTA_FP"]
    )
    res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
    res["RHOTA"] = np.sqrt(res["DetRe"] * res["AssA"])
    res["HOTA(0)"] = np.sqrt(res["DetA"] * res["AssA"])[0]
    res["LocA(0)"] = res["LocA"][0]
    res["HOTALocA(0)"] = res["HOTA(0)"] * res["LocA(0)"]

    # Calculate final scores
    hota_final_scores(res)
    return res

def hota_final_scores(res):
    """Calculate final HOTA scores"""
    res["HOTA"] = np.mean(res["HOTA"])
    res["DetA"] = np.mean(res["DetA"])
    res["AssA"] = np.mean(res["AssA"])
    res["DetRe"] = np.mean(res["DetRe"])
    res["DetPr"] = np.mean(res["DetPr"])
    res["AssRe"] = np.mean(res["AssRe"])
    res["AssPr"] = np.mean(res["AssPr"])
    res["LocA"] = np.mean(res["LocA"])
    res["RHOTA"] = np.mean(res["RHOTA"])
    res["HOTA_TP"] = np.mean(res["HOTA_TP"])
    res["HOTA_FP"] = np.mean(res["HOTA_FP"])
    res["HOTA_FN"] = np.mean(res["HOTA_FN"])
    