from __future__ import annotations

import sys
from typing import Any, Optional

import numpy as np
import pandas as pd

from soccertrack.dataframe import BBoxDataFrame

X_INDEX = 0  # xmin
Y_INDEX = 1  # ymin
W_INDEX = 2  # width
H_INDEX = 3  # height
CONFIDENCE_INDEX = 4
CLASS_ID_INDEX = 5
IMAGE_NAME_INDEX = 6


def _getArea(box: list[int]) -> int:
    """Return area of box.

    Args:
        box (list[int]): box of object

    Returns:
        area (int): area of box
    """

    area = (box[2] - box[0]) * (box[3] - box[1])
    return area


def _boxesIntersect(boxA: list[int], boxB: list[int]) -> bool:
    """Checking the position of two boxes.

    Args:
        boxA (list[int]): box of object
        boxB (list[int]): box of object

    Returns:
        bool: True if boxes intersect, False otherwise
    """

    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True


def _getIntersectionArea(boxA: list[int], boxB: list[int]) -> int:
    """Return intersection area of two boxes.

    Args:
        boxA (list[int]): box of object
        boxB (list[int]): box of object

    Returns:
        intersection_area (int): area of intersection
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    intersection_area = (xB - xA) * (yB - yA)
    # intersection area
    return intersection_area


def _getUnionAreas(
    boxA: list[int], boxB: list[int], interArea: Optional[float] = None
) -> float:
    area_A = _getArea(boxA)
    area_B = _getArea(boxB)
    if interArea is None:
        interArea = _getIntersectionArea(boxA, boxB)
    return float(area_A + area_B - interArea)


# 11-point interpolated average precision
def ElevenPointInterpolatedAP(rec: Any, prec: Any) -> list[Any]:
    """Calculate 11-point interpolated average precision.

    Args:
        rec (np.ndarray[np.float64]): recall array
        prec (np.ndarray[np.float64]): precision array

    Returns:
        Interp_ap_info (list[Any]): List containing information necessary for ap calculation
    """
    mrec = []
    # mrec.append(0)
    for e in rec:
        mrec.append(e)
    mpre = []
    for e in prec:
        mpre.append(e)
    # [mpre.append(e) for e in prec]
    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp = []
    recallValid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recallValues:
        # Obtain all recall values higher or equal than r
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        # If there are recalls above r
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min() :])
        recallValid.append(r)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / 11
    # Generating values for the plot

    Interp_ap_info = [ap, rhoInterp, recallValid, None]
    return Interp_ap_info


def iou_score(bbox_det: list[int], bbox_gt: list[int]) -> float:
    """Calculate iou between two bbox of shape (x1, y1, x2, y2).

    Args:
        list[int]: bbox of detected object.
        list[int]: bbox of ground truth object

    Returns:
        iou(float): iou_score between two bbox
    """

    # add large number to avoid bboxes outside of image
    E = 1000000
    bbox_det = [elem + E for elem in bbox_det]
    bbox_gt = [elem + E for elem in bbox_gt]
    # if boxes dont intersect
    if _boxesIntersect(bbox_det, bbox_gt) is False:
        return 0
    interArea = _getIntersectionArea(bbox_det, bbox_gt)
    union = _getUnionAreas(bbox_det, bbox_gt, interArea=interArea)
    # intersection over union
    iou = interArea / union
    assert iou >= 0
    return iou


def convert_to_x1y1x2y2(bbox: list[int]) -> list[int]:
    """Convert bbox to x1y1x2y2 format."""
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0] + bbox[2]
    y2 = bbox[1] + bbox[3]
    return [x1, y1, x2, y2]


def convert_bboxes(
    bboxes: pd.DataFrame | BBoxDataFrame | list | tuple,
) -> list[float, float, float, float, float, str, str]:
    """Convert bboxes to tuples of (xmin, ymin, width, height, confidence, class_id, image_name).

    Args:
        bboxes (pd.DataFrame | BBoxDataFrame | list | tuple): bboxes to convert.

    Returns:
        list[float, float, float, float, float, str, str]: converted bboxes.
    """

    if isinstance(bboxes, pd.DataFrame) or isinstance(bboxes, BBoxDataFrame):
        bboxes = bboxes.values.tolist()
    elif isinstance(bboxes, list):
        bboxes = [tuple(bbox) for bbox in bboxes]

    validate_bboxes(bboxes)
    return bboxes


def validate_bboxes(
    bboxes: list[float, float, float, float, float, str, str], is_gt=False
) -> None:
    for bbox in bboxes:
        assert (
            len(bbox) == 7
        ), f"bbox must have 7 elements (xmin, ymin, width, height, confidence, class_id, image_name), but {len(bbox)} elements found."

        assert isinstance(
            bbox[0], (int, float)
        ), f"xmin must be int or float, but {type(bbox[0])} found."
        assert isinstance(
            bbox[1], (int, float)
        ), f"ymin must be int or float, but {type(bbox[1])} found."
        assert isinstance(
            bbox[2], (int, float)
        ), f"width must be int or float, but {type(bbox[2])} found."
        assert isinstance(
            bbox[3], (int, float)
        ), f"height must be int or float, but {type(bbox[3])} found."
        if is_gt:
            assert (
                bbox[4] == 1
            ), f"confidence must be 1 for ground truth bbox, but {bbox[4]} found."
        else:
            assert isinstance(
                bbox[4], (int, float)
            ), f"confidence must be int or float, but {type(bbox[4])} found."
        assert isinstance(
            bbox[5], (str)
        ), f"class_id must be str, but {type(bbox[5])} found."
        assert isinstance(
            bbox[6], (str)
        ), f"image_name must be str, but {type(bbox[6])} found."


def ap_score(
    bboxes_det_per_class: list[list[float, float, float, float, float, str, str]],
    bboxes_gt_per_class: list[list[float, float, float, float, float, str, str]],
    iou_threshold: float,
) -> dict[str, Any]:
    """Calculate average precision.

    Args:
        bboxes_det_per_class(list): bbox of detected object per class.
        bboxes_gt_per_class(list): bbox of ground truth object per class.
        IOUThreshold(float): iou threshold. it is usually set to 50%, 75% or 95%.

    Returns:
        ap(dict): dict containing information about average precision

    Note:
        bboxes_det_per_class: [bbox_det_1, bbox_det_2, ...]
        bboxes_gt_per_class: [bbox_gt_1, bbox_gt_2, ...]

        #The elements of each bbox variable are as follows, each element basically corresponding to a property of the BoundingBox class of Object-Detection-Metrics.
        https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/BoundingBox.py

        ----
        bbox_det_n(tuple): (xmin, ymin, width, height, confidence, class_id, image_name)
        bbox_gt_n(tuple): (xmin, ymin, width, height, 1.0, class_id, image_name)

        xmin(float): xmin
        ymin(float): ymin
        width(float): width
        height(float): height
        confidence(float): class confidence
        class_id(str): class id
        image_name(str): image name

        #index variable, this is written as a global variable in the `def main()` function.
        X_INDEX = 0
        Y_INDEX = 1
        W_INDEX = 2
        H_INDEX = 3
        CONFIDENCE_INDEX = 4
        CLASS_ID_INDEX = 5
        IMAGE_NAME_INDEX = 6
    """
    assert len(bboxes_gt_per_class) != 0, f"It must contain at least one Grand Truth."

    class_id = bboxes_gt_per_class[0][CLASS_ID_INDEX]
    n_dets = len(bboxes_det_per_class)
    n_gts = len(bboxes_gt_per_class)

    if len(bboxes_det_per_class) == 0:
        return {
            "class": class_id,
            "precision": [],
            "recall": [],
            "AP": 0.0,
            "interpolated precision": [],
            "interpolated recall": [],
            "total positives": 0,
            "total TP": 0,
            "total FP": 0,
        }

    validate_bboxes(bboxes_det_per_class, is_gt=False)
    validate_bboxes(bboxes_gt_per_class, is_gt=True)

    # check that class_id is the same for all bboxes
    for bbox_det in bboxes_det_per_class:
        assert (
            bbox_det[CLASS_ID_INDEX] == class_id
        ), f"class_id must be the same for all bboxes, but {bbox_det[CLASS_ID_INDEX]} found."
    for bbox_gt in bboxes_gt_per_class:
        assert (
            bbox_gt[CLASS_ID_INDEX] == class_id
        ), f"class_id must be the same for all bboxes, but {bbox_gt[CLASS_ID_INDEX]} found."

    # create dictionary with bbox_gts for each image
    # s.t. gts = {image_name_1: [bbox_gt_1, bbox_gt_2, ...], image_name_2: [bbox_gt_1, bbox_gt_2, ...], ...}
    gts: dict[str, Any] = {}
    for bbox_gt in bboxes_gt_per_class:
        image_name = bbox_gt[IMAGE_NAME_INDEX]
        gts[image_name] = gts.get(image_name, []) + [bbox_gt]

    # Sort detections by decreasing confidence
    bboxes_det_per_class = sorted(
        bboxes_det_per_class, key=lambda x: x[CONFIDENCE_INDEX], reverse=True
    )

    # create dictionary with amount of gts for each image
    det = {key: np.zeros(len(gt)) for key, gt in gts.items()}

    iouMax_list = []

    # Loop through detections
    TP = np.zeros(len(bboxes_det_per_class))
    FP = np.zeros(len(bboxes_det_per_class))
    for d, bbox_det in enumerate(bboxes_det_per_class):

        # Find ground truth image
        image_name = bbox_det[IMAGE_NAME_INDEX]
        gt_bboxes = gts.get(image_name, [])
        iouMax = sys.float_info.min

        bbox_det = [
            bbox_det[X_INDEX],
            bbox_det[Y_INDEX],
            bbox_det[W_INDEX],
            bbox_det[H_INDEX],
        ]
        bbox_det = convert_to_x1y1x2y2(bbox_det)

        for j, bbox_gt in enumerate(gt_bboxes):
            bbox_gt = [
                bbox_gt[X_INDEX],
                bbox_gt[Y_INDEX],
                bbox_gt[W_INDEX],
                bbox_gt[H_INDEX],
            ]

            # convert x,y,w,h to x1,y1,x2,y2
            bbox_gt = convert_to_x1y1x2y2(bbox_gt)

            iou = iou_score(bbox_det, bbox_gt)

            if iou > iouMax:
                iouMax = iou
                jmax = j
                iouMax_list.append(iouMax)

        # Assign detection as true positive/don't care/false positive
        if iouMax >= iou_threshold:
            TP[d] = 1  # count as true positive
            det[image_name][jmax] = 1  # flag as already 'seen'
        else:
            FP[d] = 1  # count as false positive

    # compute precision, recall and average precision
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    rec = acc_TP / n_dets
    prec = np.divide(acc_TP, (acc_FP + acc_TP))

    # Depending on the method, call the right implementation
    [ap_, mpre_, mrec_, _] = ElevenPointInterpolatedAP(rec, prec)

    return {
        "class": class_id,
        "precision": list(prec),
        "recall": list(rec),
        "AP": ap_,
        "interpolated precision": mpre_,
        "interpolated recall": mrec_,
        "total positives": n_dets,
        "total TP": int(np.sum(TP)),
        "total FP": int(np.sum(FP)),
    }


def ap_score_range(
    bboxes_det_per_class: list[float, float, float, float, float, str, str],
    bboxes_gt_per_class: list[float, float, float, float, float, str, str],
    start_threshold: float = 0.5,
    end_threshold: float = 0.95,
    step: float = 0.05,
) -> float:

    """Calculate average precision in the specified range.

    Args:
        bboxes_det_per_class(list): bbox of detected object per class.
        bboxes_gt_per_class(list): bbox of ground truth object per class.
        start_threshold(float): start threshold of IOU. default is 0.5.
        end_threshold(float): end threshold of IOU. default is 0.95.
        step(float): step of updating threshold. default is 0.05.

    Returns:
        ap_results(list): list of average precision in the specified range.
        ap_range(float): average of ap in the specified range.

    """

    ap_list = []
    for iou_threshold in np.arange(start_threshold, end_threshold + step, step):
        ap_result = ap_score(bboxes_det_per_class, bboxes_gt_per_class, iou_threshold)
        ap_list.append(ap_result["AP"])

    ap_range = np.mean(ap_list)

    return ap_range


def map_score(
    bboxes_det: pd.DataFrame | BBoxDataFrame | list | tuple,
    bboxes_gt: pd.DataFrame | BBoxDataFrame | list | tuple,
    iou_threshold: float,
) -> float:
    """Calculate mean average precision.

    Args:
        det_df(pd.DataFrame): dataframe of detected object.
        gt_df(pd.DataFrame): dataframe of ground truth object.
        IOUThreshold(float): iou threshold

    Returns:
        map(float): mean average precision
    """

    # convert to 2-dim list from df
    bboxes_det = convert_bboxes(bboxes_det)
    bboxes_gt = convert_bboxes(bboxes_gt)

    ap_list = []
    class_list = []

    # calculate ap
    for bbox_gt in bboxes_gt:
        if bbox_gt[CLASS_ID_INDEX] not in class_list:
            class_list.append(bbox_gt[CLASS_ID_INDEX])

    classes = sorted(class_list)
    for class_id in classes:
        bboxes_det_per_class = [
            detection_per_class
            for detection_per_class in bboxes_det
            if detection_per_class[CLASS_ID_INDEX] == class_id
        ]
        bboxes_gt_per_class = [
            groundTruth_per_class
            for groundTruth_per_class in bboxes_gt
            if groundTruth_per_class[CLASS_ID_INDEX] == class_id
        ]
        ap = ap_score(bboxes_det_per_class, bboxes_gt_per_class, iou_threshold)
        ap_list.append(ap["AP"])

    # calculate map
    map = np.mean(ap_list)
    return map


def map_score_range(
    bboxes_det: pd.DataFrame | BBoxDataFrame | list | tuple,
    bboxes_gt: pd.DataFrame | BBoxDataFrame | list | tuple,
    start_threshold: float = 0.5,
    end_threshold: float = 0.95,
    step: float = 0.05,
) -> float:
    """Calculate mean average precision.

    Args:
        det_df(pd.DataFrame): dataframe of detected object.
        gt_df(pd.DataFrame): dataframe of ground truth object.
        start_threshold(float): start threshold of IOU. default is 0.5.
        end_threshold(float): end threshold of IOU. default is 0.95.
        step(float): step of updating threshold. default is 0.05.

    Returns:
        map_range(float): average of map in the specified range. (0.5 to 0.95 in increments of 0.05)

    """
    map_list = []
    for iou_threshold in np.arange(start_threshold, end_threshold + step, step):
        map_result = map_score(bboxes_det, bboxes_gt, iou_threshold)
        map_list.append(map_result)

    map_range = np.mean(map_list)

    return map_range
