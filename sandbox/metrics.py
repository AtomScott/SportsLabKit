import numpy as np
import sys
import random
from typing import List, Optional, Tuple
from utils import _getArea, _boxesIntersect, _getIntersectionArea, _getUnionAreas, ElevenPointInterpolatedAP

### Object detection metrics ###

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

def ap_score(bboxes_det_per_class, bboxes_gt_per_class, IOUThreshold ,ap_only):
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
        bbox_det_n(tuple): (imageName, classId, classConfidence, [xmin, ymin, xmax, ymax])
        bbox_gt_n(tuple): (imageName, classId, 1.0, [xmin, ymin, xmax, ymax])
        imageName(str): image name
        classId(str): class id
        classConfidence(float): class confidence
        xmin(float): xmin
        ymin(float): ymin
        xmax(float): xmax
        ymax(float): ymax


    """
    iouMax_list = []
    gts = {}
    npos = 0
    for g in bboxes_gt_per_class:
        npos += 1
        gts[g[0]] = gts.get(g[0], []) + [g]

    # sort detections by decreasing confidence
    dects = sorted(bboxes_det_per_class, key=lambda conf: conf[2], reverse=True)
    # print("dects: ", dects)
    # create dictionary with amount of gts for each image
    det = {key: np.zeros(len(gts[key])) for key in gts}
    # print(dects)

    print("Evaluating class: %s (%d detections)" % (str(dects[0][1]), len(dects)))
    # Loop through detections
    TP = np.zeros(len(dects))
    FP = np.zeros(len(dects))
    for d in range(len(dects)):
        # print('dect %s => %s' % (dects[d][0], dects[d][3],))
        # Find ground truth image
        gt = gts[dects[d][0]] if dects[d][0] in gts else []
        iouMax = sys.float_info.min

        for j in range(len(gt)):
            iou = iou_score(dects[d][3], gt[j][3])
            if iou > iouMax:
                iouMax = iou
                jmax = j
                iouMax_list.append(iouMax)

        # Assign detection as true positive/don't care/false positive
        if iouMax >= IOUThreshold:
            TP[d] = 1  # count as true positive
            det[dects[d][0]][jmax] = 1  # flag as already 'seen'
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
        ap = {
            'class': dects[0][1],
            'AP': ap_
            }
        return ap
    else:
        ap = {
            'class': dects[0][1],
            'precision': prec,
            'recall': rec,
            'AP': ap_,
            'interpolated precision': mpre_,
            'interpolated recall': mrec_,
            'total positives': npos,
            'total TP': np.sum(TP),
            'total FP': np.sum(FP)
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
    ap_list = []
    class_list = []
    #calculate ap
    for bbox_det in bboxes_det:
        if bbox_det[1] not in class_list:
            class_list.append(bbox_det[1])

    classes = sorted(class_list)
    for class_id in classes:
        bboxes_det_per_class = [detection_per_class for detection_per_class in bboxes_det if detection_per_class[1] == class_id]
        bboxes_gt_per_class = [groundTruth_per_class for groundTruth_per_class in bboxes_gt if groundTruth_per_class[1] == class_id]
        ap = ap_score(bboxes_det_per_class, bboxes_gt_per_class, IOUThreshold, ap_only=True)
        # print(f'ap: {ap}')
        ap_list.append(ap['AP'])
    #calculate map
    map = np.mean(ap_list)
    return map

### Object detection metrics ###



def main():#(動作確認用)
    #create dummy bounding box
    image_num = 5 #number of images
    bbox_offset = 0 #検出用バウンディングボックスのオフセット
    IOUThreshold = 0.5 #IOU閾値
    ap_only = True #calc_ap関数で、APのみを出力するかどうかのトリガー

    bboxes_det = []
    bboxes_gt = []
    classes = []

    for image_name in range(image_num):
        for ClassId in [0 ,32] :
            for num in range(0, 1000, 100):
                imageName = f'{image_name}'
                classId = f'{ClassId}'

                bbox_offset = random.randint(0, 100) #Offset of the bounding box(random) <- ハイパラ
                # print(f'bbox_offset: {bbox_offset}')
                

                num_gt = num
                num_det = num + bbox_offset
                # print(f'image_name: {getImageName} , class_id: {getClassId} , num_gt: {num_gt}, num_det: {num_det}')

                x1 = (1 * num_gt , 1 * num_det)
                y1 = (1 * num_gt , 1 * num_det)
                x2 = (1 * num_gt + 50 , 1 * num_det+ 50)
                y2 = (1 * num_gt + 50 , 1 * num_det+ 50)


                getAbsoluteBoundingBox_det = (x1[1], y1[1], x2[1], y2[1])
                getAbsoluteBoundingBox_gt = (x1[0], y1[0], x2[0], y2[0])
                
                
                getConfidence = random.random()

                gt_info = [imageName, classId, 1, getAbsoluteBoundingBox_gt]
                det_info = [imageName, classId, getConfidence ,getAbsoluteBoundingBox_det]

                if classId not in classes:
                    classes.append(classId)

                bboxes_det.append(det_info)
                bboxes_gt.append(gt_info)

    #### test evaluation ###
    #calculate iou
    for i in range(len(bboxes_det)):
        iou = iou_score(bboxes_det[i][3], bboxes_gt[i][3])
        # print(f'iou: {iou}')

    #calculate ap
    classes = sorted(classes)
    for class_id in classes:
        bboxes_det_per_class = [detection_per_class for detection_per_class in bboxes_det if detection_per_class[1] == class_id]
        bboxes_gt_per_class = [groundTruth_per_class for groundTruth_per_class in bboxes_gt if groundTruth_per_class[1] == class_id]
        ap = ap_score(bboxes_det_per_class, bboxes_gt_per_class, IOUThreshold, ap_only=True)
        print(f'ap: {ap}')

    print('----------------------------------------------------')
    #calculate map
    map = map_score(bboxes_gt, bboxes_det, IOUThreshold)
    print(f'map: {map}')