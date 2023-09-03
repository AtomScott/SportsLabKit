import numpy as np
from scipy.spatial.distance import cdist

from sportslabkit.logger import tqdm
from sportslabkit.metrics import iou_score


def list2dict(bboxes_track, bboxes_gt):
    """[frame, object_id, 1 ,[x_min, y_min, w, h]]"""
    data = {}

    split_ids_track = []
    for i in range(len(bboxes_track)):
        try:
            if bboxes_track[i][0] != bboxes_track[i + 1][0]:
                split_ids_track.append(i + 1)
            else:
                pass
        except IndexError:
            pass

    split_ids_gt = []
    for i in range(len(bboxes_gt)):
        try:
            if bboxes_gt[i][0] != bboxes_gt[i + 1][0]:
                split_ids_gt.append(i + 1)
            else:
                pass
        except IndexError:
            pass

    split_data_track = np.split(bboxes_track, split_ids_track)
    split_data_gt = np.split(bboxes_gt, split_ids_gt)

    tracker_ids = []
    tracker_dets = []
    for split_elem in split_data_track:
        tracker_dets_TEMP = []
        tracker_id = np.asarray(split_elem[:, 1], dtype="int64")
        tracker_ids.append(tracker_id)
        for box in split_elem[:, 3]:
            tracker_dets_TEMP.append(box)
        tracker_dets.append(np.array(tracker_dets_TEMP))

    gt_ids = []
    gt_dets = []
    for split_elem in split_data_gt:
        gt_dets_TEMP = []
        gt_id = np.asarray(split_elem[:, 1], dtype="int64")
        gt_ids.append(gt_id)
        for box in split_elem[:, 3]:
            gt_dets_TEMP.append(box)
        gt_dets.append(np.array(gt_dets_TEMP))

    gt_dets_xyxy = []
    for gt_det in gt_dets:
        gt_det[:, 2] = gt_det[:, 2] + gt_det[:, 0]
        gt_det[:, 3] = gt_det[:, 3] + gt_det[:, 1]
        gt_dets_xyxy.append(gt_det)

    track_dets_xyxy = []
    for track_det in tracker_dets:
        track_det[:, 2] = track_det[:, 2] + track_det[:, 0]
        track_det[:, 3] = track_det[:, 3] + track_det[:, 1]
        track_dets_xyxy.append(track_det)

    sim_score_list = []
    for i in tqdm(range(len(gt_ids))):
        sim_score = cdist(gt_dets_xyxy[i], track_dets_xyxy[i], iou_score)
        sim_score_list.append(sim_score)

    num_tracker_dets = sum([len(tracker_dets[i]) for i in range(len(tracker_dets))])
    num_gt_dets = sum([len(gt_dets[i]) for i in range(len(gt_dets))])

    data = {}
    data["gt_ids"] = gt_ids
    data["tracker_ids"] = tracker_ids
    data["tracker_dets"] = tracker_dets
    data["gt_dets"] = gt_dets
    data["similarity_scores"] = sim_score_list
    data["num_tracker_dets"] = num_tracker_dets
    data["num_gt_dets"] = num_gt_dets
    data["num_tracker_ids"] = 22
    data["num_gt_ids"] = 22
    data["num_timesteps"] = 100
    data["seq"] = "test"
    return data
