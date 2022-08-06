import numpy as np
import sys
import random
from typing import List, Optional, Tuple
from metrics import iou_score, ap_score, map_score


#create dummy bounding box
X1_INDEX = 0
Y1_INDEX = 1
X2_INDEX = 2
Y2_INDEX = 3
CONFIDENCE_INDEX = 4
CLASS_ID_IMDEX = 5
IMAGE_NAME_INDEX = 6

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
            class_id = f'{ClassId}'

            bbox_offset = random.randint(0, 100) #Offset of the bounding box(random) <- ハイパラ
            # print(f'bbox_offset: {bbox_offset}')
            

            num_gt = num
            num_det = num + bbox_offset
            # print(f'image_name: {getImageName} , class_id: {getClassId} , num_gt: {num_gt}, num_det: {num_det}')

            x1 = (1 * num_gt , 1 * num_det)
            y1 = (1 * num_gt , 1 * num_det)
            x2 = (1 * num_gt + 50 , 1 * num_det+ 50)
            y2 = (1 * num_gt + 50 , 1 * num_det+ 50)


            # getAbsoluteBoundingBox_gt = (x1[0], y1[0], x2[0], y2[0])
            # getAbsoluteBoundingBox_det = (x1[1], y1[1], x2[1], y2[1])
            
            confidence = random.random()

            gt_info = (x1[0], y1[0], x2[0], y2[0], 1, class_id, image_name)
            det_info = [x1[1], y1[1], x2[1], y2[1], confidence, class_id, image_name]

            if class_id not in classes:
                classes.append(class_id)

            bboxes_det.append(det_info)
            bboxes_gt.append(gt_info)

#### test evaluation ###
#calculate iou
for i in range(len(bboxes_det)):
    bbox_det = [bboxes_det[i][X1_INDEX], bboxes_det[i][Y1_INDEX], bboxes_det[i][X2_INDEX], bboxes_det[i][Y2_INDEX]]
    bbox_gt = [bboxes_gt[i][X1_INDEX], bboxes_gt[i][Y1_INDEX], bboxes_gt[i][X2_INDEX], bboxes_gt[i][Y2_INDEX]]
    iou = iou_score(bbox_det, bbox_gt)

#calculate ap
classes = sorted(classes)
for class_id in classes:
    bboxes_det_per_class = [detection_per_class for detection_per_class in bboxes_det if detection_per_class[CLASS_ID_IMDEX] == class_id]
    bboxes_gt_per_class = [groundTruth_per_class for groundTruth_per_class in bboxes_gt if groundTruth_per_class[CLASS_ID_IMDEX] == class_id]
    ap = ap_score(bboxes_det_per_class, bboxes_gt_per_class, IOUThreshold, ap_only=True)
    print(f'ap: {ap}')

print('----------------------------------------------------')
#calculate map
map = map_score(bboxes_det, bboxes_gt, IOUThreshold)
print(f'map: {map}')

