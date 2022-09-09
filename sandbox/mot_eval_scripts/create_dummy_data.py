import numpy as np
import sys
import random
from typing import List, Optional, Tuple
from scipy.spatial.distance import cdist
from tqdm import tqdm

def _getArea(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def _boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True

def _getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA) * (yB - yA)

def _getUnionAreas(boxA, boxB, interArea=None):
    area_A = _getArea(boxA)
    area_B = _getArea(boxB)
    if interArea is None:
        interArea = _getIntersectionArea(boxA, boxB)
    return float(area_A + area_B - interArea)

def iou_score(bbox_det: tuple, bbox_gt: tuple) -> float:
    """calculate iou between two bbox
    Args:
        bbox_det(tuple): bbox of detected object. 
        bbox_gt(tuple): bbox of ground truth object

    Returns:
        iou(float): iou_score between two bbox

    Note:

    """
    # if boxes dont intersect
    if _boxesIntersect(bbox_det, bbox_gt) is False:
        return 0
    interArea = _getIntersectionArea(bbox_det, bbox_gt)
    union = _getUnionAreas(bbox_det, bbox_gt, interArea=interArea)
    # intersection over union
    iou = interArea / union
    assert iou >= 0
    return iou

#create dummy bounding box
def create_raw_dummy_data():
    bboxes_track = []
    bboxes_gt = []
    classes = []

    num_gt = 0
    num_track = 0

    offset = 0
    position_trans = 50
    frame_trans = 10

    for frame in range(100):
        num_gt_ = num_gt
        num_track_ = num_track
        for object_id in range(22):
            object_id = np.int64(object_id)
                
            x1 = (1 * num_gt_ , 1 * num_track_)
            y1 = (1 * num_gt_ , 1 * num_track_)
            w = 50
            h = 50

            num_gt_ += position_trans
            num_track_ += position_trans + offset
            
            getAbsoluteBoundingBox_track = [x1[1], y1[1], w, h]
            getAbsoluteBoundingBox_gt = [x1[0], y1[0], w, h]
            
            getConfidence = random.random()

            track_info = [frame, object_id, 1 ,getAbsoluteBoundingBox_track]
            gt_info = [frame, object_id, 1, getAbsoluteBoundingBox_gt]
            
            bboxes_track.append(track_info)
            bboxes_gt.append(gt_info)
        num_gt += frame_trans
        num_track += frame_trans + offset
    return bboxes_track, bboxes_gt

def create_dummy_data(bboxes_track, bboxes_gt):
    data = {}

    split_ids_track=[]
    for i in range(len(bboxes_track)):
        try:
            if bboxes_track[i][0] != bboxes_track[i+1][0]:
                split_ids_track.append(i+1)
            else:
                pass
        except IndexError:
            pass

    split_ids_gt=[]
    for i in range(len(bboxes_gt)):
        try:
            if bboxes_gt[i][0] != bboxes_gt[i+1][0]:
                split_ids_gt.append(i+1)
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
        tracker_id = np.asarray(split_elem[:,1],dtype='int64')
        tracker_ids.append(tracker_id)
        for box in split_elem[:,3]:
            tracker_dets_TEMP.append(box)
        tracker_dets.append(np.array(tracker_dets_TEMP))

    gt_ids = []
    gt_dets = []
    for split_elem in split_data_gt:
        gt_dets_TEMP = []
        gt_id = np.asarray(split_elem[:,1],dtype='int64')
        gt_ids.append(gt_id)
        for box in split_elem[:,3]:
            gt_dets_TEMP.append(box)
        gt_dets.append(np.array(gt_dets_TEMP))

    gt_dets_xyxy = []
    for gt_det in gt_dets:
        gt_det[:,2] = gt_det[:,2] + gt_det[:,0]
        gt_det[:,3] = gt_det[:,3] + gt_det[:,1]
        gt_dets_xyxy.append(gt_det)

    track_dets_xyxy = []
    for track_det in tracker_dets:
        track_det[:,2] = track_det[:,2] + track_det[:,0]
        track_det[:,3] = track_det[:,3] + track_det[:,1]
        track_dets_xyxy.append(track_det)

    sim_score_list = []
    for i in tqdm(range(len(gt_ids))):
        sim_score = cdist(gt_dets_xyxy[i], track_dets_xyxy[i],  iou_score)
        sim_score_list.append(sim_score)

    num_tracker_dets = sum([len(tracker_dets[i]) for i in range(len(tracker_dets))])
    num_gt_dets = sum([len(gt_dets[i]) for i in range(len(gt_dets))])

    data = {}
    data['gt_ids'] = gt_ids
    data['tracker_ids'] = tracker_ids
    data['tracker_dets'] = tracker_dets
    data['gt_dets'] = gt_dets
    data['similarity_scores'] = sim_score_list
    data['num_tracker_dets'] = num_tracker_dets
    data['num_gt_dets'] = num_gt_dets
    data['num_tracker_ids'] = 22
    data['num_gt_ids'] = 22
    data['num_timesteps'] = 100
    data['seq'] = 'test'
    return data