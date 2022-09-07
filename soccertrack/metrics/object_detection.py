import sys

import numpy as np


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


# 11-point interpolated average precision
def ElevenPointInterpolatedAP(rec, prec):
    # def CalculateAveragePrecision2(rec, prec):
    mrec = []
    # mrec.append(0)
    [mrec.append(e) for e in rec]
    # mrec.append(1)
    mpre = []
    # mpre.append(0)
    [mpre.append(e) for e in prec]
    # mpre.append(0)
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
    rvals = []
    rvals.append(recallValid[0])
    [rvals.append(e) for e in recallValid]
    rvals.append(0)
    pvals = []
    pvals.append(0)
    [pvals.append(e) for e in rhoInterp]
    pvals.append(0)
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i in range(len(rvals)):
        p = (rvals[i], pvals[i - 1])
        if p not in cc:
            cc.append(p)
        p = (rvals[i], pvals[i])
        if p not in cc:
            cc.append(p)
    recallValues = [i[0] for i in cc]
    rhoInterp = [i[1] for i in cc]
    return [ap, rhoInterp, recallValues, None]


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


def ap_score(bboxes_det_per_class, bboxes_gt_per_class, IOUThreshold, ap_only):
    """calculate average precision

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
    CLASS_ID_IMDEX = 5
    IMAGE_NAME_INDEX = 6

    iouMax_list = []
    gts = {}
    npos = 0
    for g in bboxes_gt_per_class:
        npos += 1
        gts[g[IMAGE_NAME_INDEX]] = gts.get(g[IMAGE_NAME_INDEX], []) + [g]

    # sort detections by decreasing confidence
    dects = sorted(bboxes_det_per_class, key=lambda conf: conf[4], reverse=True)

    # print("dects: ", dects)
    # create dictionary with amount of gts for each image
    det = {key: np.zeros(len(gts[key])) for key in gts}

    print(
        "Evaluating class: %s (%d detections)"
        % (str(dects[0][CLASS_ID_IMDEX]), len(dects))
    )
    # Loop through detections
    TP = np.zeros(len(dects))
    FP = np.zeros(len(dects))
    for d in range(len(dects)):
        # print('dect %s => %s' % (dects[d][0], dects[d][3],))
        # Find ground truth image
        gt = (
            gts[dects[d][IMAGE_NAME_INDEX]] if dects[d][IMAGE_NAME_INDEX] in gts else []
        )
        iouMax = sys.float_info.min

        for j in range(len(gt)):
            bbox_det = [
                dects[d][X1_INDEX],
                dects[d][Y1_INDEX],
                dects[d][X2_INDEX],
                dects[d][Y2_INDEX],
            ]
            bbox_gt = [
                gt[j][X1_INDEX],
                gt[j][Y1_INDEX],
                gt[j][X2_INDEX],
                gt[j][Y2_INDEX],
            ]
            iou = iou_score(bbox_det, bbox_gt)
            if iou > iouMax:
                iouMax = iou
                jmax = j
                iouMax_list.append(iouMax)

        # Assign detection as true positive/don't care/false positive
        if iouMax >= IOUThreshold:
            TP[d] = 1  # count as true positive
            det[dects[d][IMAGE_NAME_INDEX]][jmax] = 1  # flag as already 'seen'
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
        ap = {"class": dects[0][CLASS_ID_IMDEX], "AP": ap_}
        return ap
    else:
        ap = {
            "class": dects[0][CLASS_ID_IMDEX],
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


def map_score(bboxes_det, bboxes_gt, IOUThreshold) -> float:
    """calculate mean average precision
    Args:
        bboxes_det(list): bbox of detected object.
        bboxes_gt(list): bbox of ground truth object
        IOUThreshold(float): iou threshold

    Returns:
        map_score(float): mean average precision
    """
    X1_INDEX = 0
    Y1_INDEX = 1
    X2_INDEX = 2
    Y2_INDEX = 3
    CONFIDENCE_INDEX = 4
    CLASS_ID_IMDEX = 5
    IMAGE_NAME_INDEX = 6
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
    return map


### Object detection metrics ###
