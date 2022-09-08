from __future__ import annotations

import sys
from typing import Any, Optional

import numpy as np


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
    intersection_area = (xB - xA + 1) * (yB - yA + 1)
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
    recallValues = recallValues[::-1]
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
    rvals = []
    rvals.append(recallValid[0])
    for e in recallValid:
        rvals.append(e)
    # [rvals.append(e) for e in recallValid]
    rvals.append(0)

    pvals = []
    pvals.append(0)
    for e in rhoInterp:
        pvals.append(e)
    # [pvals.append(e) for e in rhoInterp]
    pvals.append(0)
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i, rval in enumerate(rvals):
        p = [rval, pvals[i - 1]]
        if p not in cc:
            cc.append(p)
        p = [rval, pvals[i]]
        if p not in cc:
            cc.append(p)
    recallValue = [i[0] for i in cc]
    # recallValues = cc[:, 0]
    rhoInter = [i[1] for i in cc]
    # rhoInterp = cc[:, 1]
    Interp_ap_info = [ap, rhoInter, recallValue, None]
    return Interp_ap_info


def iou_score(bbox_det: list[int], bbox_gt: list[int]) -> float:
    """Calculate iou between two bbox.

    Args:
        list[int]: bbox of detected object.
        list[int]: bbox of ground truth object

    Returns:
        iou(float): iou_score between two bbox

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


def ap_score(
    bboxes_det_per_class: list[Any],
    bboxes_gt_per_class: list[Any],
    IOUThreshold: float,
    ap_only: bool,
) -> dict[str, Any]:
    """Calculate average precision.

    Args:
        bboxes_det_per_class(list): bbox of detected object per class.
        bboxes_gt_per_class(list): bbox of ground truth object per class.
        IOUThreshold(float): iou threshold
        ap_only(bool): if True, return ap only. if False, return ap ,recall, precision, and so on.
    Returns:
        ap(float): average precision

    Note:
        bboxes_det_per_class: [bbox_det_1, bbox_det_2, ...]
        bboxes_gt_per_class: [bbox_gt_1, bbox_gt_2, ...]

        #The elements of each bbox variable are as follows, each element basically corresponding to a property of the BoundingBox class of Object-Detection-Metrics.
        https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/BoundingBox.py

        ----
        bbox_det_n(tuple): (xmin, ymin, xmax, ymax, confidence, class_id, image_name)
        bbox_gt_n(tuple): (xmin, ymin, xmax, ymax, 1.0, class_id, image_name)

        xmin(float): xmin
        ymin(float): ymin
        xmax(float): xmax
        ymax(float): ymax
        confidence(float): class confidence
        class_id(str): class id
        image_name(str): image name

        #index variable, this is written as a global variable in the `def main()` function.
        X1_INDEX = 0
        Y1_INDEX = 1
        X2_INDEX = 2
        Y2_INDEX = 3
        CONFIDENCE_INDEX = 4
        CLASS_ID_IMDEX = 5
        IMAGE_NAME_INDEX = 6

    """
    X1_INDEX = 0
    Y1_INDEX = 1
    X2_INDEX = 2
    Y2_INDEX = 3
    CONFIDENCE_INDEX = 4
    CLASS_ID_INDEX = 5
    IMAGE_NAME_INDEX = 6

    iouMax_list = []
    gts: dict[str, Any] = {}
    npos = 0
    for g in bboxes_gt_per_class:
        npos += 1
        gts[g[IMAGE_NAME_INDEX]] = gts.get(g[IMAGE_NAME_INDEX], []) + [g]
    # print(gts)

    def sort_key(x: list[Any]) -> Any:
        """Sort key.

        Args:
            x(list): bbox of detected object per class.

        Returns:
            confidence(float): confidence of bbox
        """
        CONFIDENCE_score = x[CONFIDENCE_INDEX]
        return CONFIDENCE_score

    dect = sorted(bboxes_det_per_class, key=sort_key, reverse=True)
    # create dictionary with amount of gts for each image
    det = {key: np.zeros(len(gt)) for key, gt in gts.items()}

    print(f"Evaluating class: {str(dect[0][CLASS_ID_INDEX])} ({len(dect)} detections)")
    # Loop through detections
    TP = np.zeros(len(dect))
    FP = np.zeros(len(dect))
    for d, dect in enumerate(dect):
        # print('dect %s => %s' % (dects[d][0], dects[d][3],))
        # Find ground truth image
        gt = gts[dect[IMAGE_NAME_INDEX]] if dect[IMAGE_NAME_INDEX] in gts else []
        iouMax = sys.float_info.min

        for j, gt_elem in enumerate(gt):
            bbox_det = [
                dect[X1_INDEX],
                dect[Y1_INDEX],
                dect[X2_INDEX],
                dect[Y2_INDEX],
            ]
            bbox_gt = [
                gt_elem[X1_INDEX],
                gt_elem[Y1_INDEX],
                gt_elem[X2_INDEX],
                gt_elem[Y2_INDEX],
            ]
            iou = iou_score(bbox_det, bbox_gt)
            if iou > iouMax:
                iouMax = iou
                jmax = j
                iouMax_list.append(iouMax)

        # Assign detection as true positive/don't care/false positive
        if iouMax >= IOUThreshold:
            TP[d] = 1  # count as true positive
            det[dect[IMAGE_NAME_INDEX]][jmax] = 1  # flag as already 'seen'
        else:
            FP[d] = 1  # count as false positive

    # compute precision, recall and average precision
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    # Depending on the method, call the right implementation
    [ap_, mpre_, mrec_, _] = ElevenPointInterpolatedAP(rec, prec)
    # if method == MethodAveragePrecision.EveryPointInterpolation:

    if ap_only:
        ap = {"class": dect[CLASS_ID_INDEX], "AP": ap_}

    else:
        ap = {
            "class": dect[CLASS_ID_INDEX],
            "precision": prec,
            "recall": rec,
            "AP": ap_,
            "interpolated precision": mpre_,
            "interpolated recall": mrec_,
            "total positives": npos,
            "total TP": np.sum(TP),
            "total FP": np.sum(FP),
        }
    return ap


def map_score(
    bboxes_det: list[Any], bboxes_gt: list[Any], IOUThreshold: float
) -> float:
    """Calculate mean average precision.

    Args:
        bboxes_det(list[Any]): bbox of detected object.
        bboxes_gt(list[Any]): bbox of ground truth object
        IOUThreshold(list[Any]): iou threshold

    Returns:
        map_score(Any): mean average precision
    """
    # X1_INDEX = 0
    # Y1_INDEX = 1
    # X2_INDEX = 2
    # Y2_INDEX = 3
    # CONFIDENCE_INDEX = 4
    CLASS_ID_IMDEX = 5
    # IMAGE_NAME_INDEX = 6
    ap_list = []
    class_list = []
    # calculate ap
    for bbox_det in bboxes_det:
        if bbox_det[CLASS_ID_IMDEX] not in class_list:
            class_list.append(bbox_det[CLASS_ID_IMDEX])

    classes = sorted(class_list)
    for class_id in classes:
        bboxes_det_per_class = [
            detection_per_class
            for detection_per_class in bboxes_det
            if detection_per_class[CLASS_ID_IMDEX] == class_id
        ]
        bboxes_gt_per_class = [
            groundTruth_per_class
            for groundTruth_per_class in bboxes_gt
            if groundTruth_per_class[CLASS_ID_IMDEX] == class_id
        ]
        ap = ap_score(
            bboxes_det_per_class, bboxes_gt_per_class, IOUThreshold, ap_only=True
        )
        print(f"ap: {ap}")
        ap_list.append(ap["AP"])
    # calculate map
    map = np.mean(ap_list)
    return float(map)


### Object detection metrics ###
