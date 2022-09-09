from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

# ###def mainを消すときは削除###
# import sys
# sys.path.append('../../')

# from sandbox.mot_eval_scripts.create_dummy_data import create_raw_dummy_data, create_dummy_data
# ######

def mota_score(data):
    """Calculates CLEAR metrics for one sequence"""
    # Initialise results
    main_integer_fields = ['CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag']
    extra_integer_fields = ['CLR_Frames']
    integer_fields = main_integer_fields + extra_integer_fields
    main_float_fields = ['MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'sMOTA']
    extra_float_fields = ['CLR_F1', 'FP_per_frame', 'MOTAL', 'MOTP_sum']
    float_fields = main_float_fields + extra_float_fields
    fields = float_fields + integer_fields

    # Configuration options:
    # config = utils.init_config(config, get_default_config(), get_name())
    threshold = 0.5
    res = {}
    for field in fields:
        res[field] = 0

    # Return result quickly if tracker or gt sequence is empty
    if data['num_tracker_dets'] == 0:
        res['CLR_FN'] = data['num_gt_dets']
        res['ML'] = data['num_gt_ids']
        res['MLR'] = 1.0
        return res
    if data['num_gt_dets'] == 0:
        res['CLR_FP'] = data['num_tracker_dets']
        res['MLR'] = 1.0
        return res

    # Variables counting global association
    num_gt_ids = data['num_gt_ids']
    gt_id_count = np.zeros(num_gt_ids)  # For MT/ML/PT
    gt_matched_count = np.zeros(num_gt_ids)  # For MT/ML/PT
    gt_frag_count = np.zeros(num_gt_ids)  # For Frag

    # Note that IDSWs are counted based on the last time each gt_id was present (any number of frames previously),
    # but are only used in matching to continue current tracks based on the gt_id in the single previous timestep.
    prev_tracker_id = np.nan * np.zeros(num_gt_ids)  # For scoring IDSW
    prev_timestep_tracker_id = np.nan * np.zeros(num_gt_ids)  # For matching IDSW

    # Calculate scores for each timestep
    for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
        # Deal with the case that there are no gt_det/tracker_det in a timestep.
        if len(gt_ids_t) == 0:
            res['CLR_FP'] += len(tracker_ids_t)
            continue
        if len(tracker_ids_t) == 0:
            res['CLR_FN'] += len(gt_ids_t)
            gt_id_count[gt_ids_t] += 1
            continue

        # Calc score matrix to first minimise IDSWs from previous frame, and then maximise MOTP secondarily
        similarity = data['similarity_scores'][t]
        score_mat = (tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]])
        score_mat = 1000 * score_mat + similarity
        score_mat[similarity < threshold - np.finfo('float').eps] = 0

        # Hungarian algorithm to find best matches
        match_rows, match_cols = linear_sum_assignment(-score_mat)
        actually_matched_mask = score_mat[match_rows, match_cols] > 0 + np.finfo('float').eps
        match_rows = match_rows[actually_matched_mask]
        match_cols = match_cols[actually_matched_mask]

        matched_gt_ids = gt_ids_t[match_rows]
        matched_tracker_ids = tracker_ids_t[match_cols]

        # Calc IDSW for MOTA
        prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
        is_idsw = (np.logical_not(np.isnan(prev_matched_tracker_ids))) & (
            np.not_equal(matched_tracker_ids, prev_matched_tracker_ids))
        res['IDSW'] += np.sum(is_idsw)

        # Update counters for MT/ML/PT/Frag and record for IDSW/Frag for next timestep
        gt_id_count[gt_ids_t] += 1
        gt_matched_count[matched_gt_ids] += 1
        not_previously_tracked = np.isnan(prev_timestep_tracker_id)
        prev_tracker_id[matched_gt_ids] = matched_tracker_ids
        prev_timestep_tracker_id[:] = np.nan
        prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids
        currently_tracked = np.logical_not(np.isnan(prev_timestep_tracker_id))
        gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)

        # Calculate and accumulate basic statistics
        num_matches = len(matched_gt_ids)
        res['CLR_TP'] += num_matches
        res['CLR_FN'] += len(gt_ids_t) - num_matches
        res['CLR_FP'] += len(tracker_ids_t) - num_matches
        if num_matches > 0:
            res['MOTP_sum'] += sum(similarity[match_rows, match_cols])
    
        # Calculate MT/ML/PT/Frag/MOTP
        tracked_ratio = gt_matched_count[gt_id_count > 0] / gt_id_count[gt_id_count > 0]
        res['MT'] = np.sum(np.greater(tracked_ratio, 0.8))
        res['PT'] = np.sum(np.greater_equal(tracked_ratio, 0.2)) - res['MT']
        res['ML'] = num_gt_ids - res['MT'] - res['PT']
        res['Frag'] = np.sum(np.subtract(gt_frag_count[gt_frag_count > 0], 1))
        res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])

        res['CLR_Frames'] = data['num_timesteps']

    # Calculate final CLEAR scores
    """Calculate sub-metric ('field') values which only depend on other sub-metric values.
    This function is used both for both per-sequence calculation, and in combining values across sequences.
    """
    num_gt_ids = res['MT'] + res['ML'] + res['PT']
    res['MTR'] = res['MT'] / np.maximum(1.0, num_gt_ids)
    res['MLR'] = res['ML'] / np.maximum(1.0, num_gt_ids)
    res['PTR'] = res['PT'] / np.maximum(1.0, num_gt_ids)
    res['CLR_Re'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
    res['CLR_Pr'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res['CLR_FP'])
    res['MODA'] = (res['CLR_TP'] - res['CLR_FP']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
    res['MOTA'] = (res['CLR_TP'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
    res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])
    res['sMOTA'] = (res['MOTP_sum'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])

    res['CLR_F1'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + 0.5*res['CLR_FN'] + 0.5*res['CLR_FP'])
    res['FP_per_frame'] = res['CLR_FP'] / np.maximum(1.0, res['CLR_Frames'])
    safe_log_idsw = np.log10(res['IDSW']) if res['IDSW'] > 0 else res['IDSW']
    res['MOTAL'] = (res['CLR_TP'] - res['CLR_FP'] - safe_log_idsw) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])

    return res

# def main():
#     #create dummy data
#     bboxes_track, bboxes_gt = create_raw_dummy_data()
#     data = create_dummy_data(bboxes_track, bboxes_gt)

#     mota = mota_score(data)
#     print(mota)

# if __name__ == '__main__':
#     main()
