import random
import numpy as np
import unittest

from soccertrack.metrics import ap_score, iou_score, map_score, ap_score_range, map_score_range


class TestMetrics(unittest.TestCase):
    def test_iou_score_1(self):
        bbox_det = [10, 10, 20, 20]
        bbox_gt = [10, 15, 20, 20]

        iou = iou_score(bbox_det, bbox_gt)
        ans = 0.5
        self.assertEqual(iou, ans)

    def test_iou_score_2(self):
        bbox_det = [10, 10, 20, 20]
        bbox_gt = [10, 10, 20, 20]

        iou = iou_score(bbox_det, bbox_gt)
        ans = 1.0
        self.assertEqual(iou, ans)

    def test_iou_score_3(self):
        bbox_det = [10, 10, 20, 20]
        bbox_gt = [0, 1, 2, 3]

        iou = iou_score(bbox_det, bbox_gt)
        ans = 0
        self.assertEqual(iou, ans)

    def test_ap_score_1(self):
        """Test AP score with perfect detection."""
        bboxes_det = [
            [10, 10, 20, 20, 1.0, "A", "N"],
            [10, 10, 20, 20, 1.0, "A", "M"],
        ]

        bboxes_gt = [
            [10, 10, 20, 20, 1, "A", "N"],
            [10, 10, 20, 20, 1, "A", "M"],
        ]

        ap = ap_score(bboxes_det, bboxes_gt, 0.5)
        ans = {
            "class": "A",
            "precision": [1., 1.],  
            "recall": [0.5, 1.],  
            "AP": 1.0,
            "interpolated precision": [1.0 for i in range(11)],  
            "interpolated recall": list(np.linspace(0, 1, 11)[::-1]), 
            "total positives": 2,
            "total TP": 2,
            "total FP": 0,
        }
        self.assertDictEqual(ap, ans)
        
    def test_ap_score_range1(self):
        """Test average AP score within specified range with perfect detection."""
        bboxes_det = [
            [10, 10, 20, 20, 1.0, "A", "N"],
            [10, 10, 20, 20, 1.0, "A", "M"],
        ]

        bboxes_gt = [
            [10, 10, 20, 20, 1, "A", "N"],
            [10, 10, 20, 20, 1, "A", "M"],
        ]

        ap_range = ap_score_range(bboxes_det, bboxes_gt, 0.5, 0.95, 0.05)
        ans = 1.0
        self.assertEqual(ap_range, ans)

    def test_ap_score_2(self):
        """Test AP score with zero detection."""
        bboxes_det = []
        bboxes_gt = [
            [10, 10, 20, 20, 1, "A", "N"],
            [10, 10, 20, 20, 1, "A", "M"],
        ]
        ap = ap_score(bboxes_det, bboxes_gt, 0.5)
        ans = {
            "class": "A",
            "precision": [],  
            "recall": [],  
            "AP": 0.0,
            "interpolated precision": [],  
            "interpolated recall": [],  
            "total positives": 0,
            "total TP": 0,
            "total FP": 0,
        }
        self.assertDictEqual(ap, ans)

    def test_ap_score_3(self):
        """Test AP score with false negative detection."""
        
        bboxes_det = [
            [10, 10, 20, 20, 1.0, "A", "N"],
            [10, 10, 20, 20, 1.0, "A", "M"],
        ]
        bboxes_gt = []

        with self.assertRaises(AssertionError):
            ap_score(bboxes_det, bboxes_gt, 0.5)

    def test_map_score_1(self):
        """Test mAP score with perfect detection using a list of 7 values as input"""
        bboxes_det = [
            [10, 10, 20, 20, 1.0, "A", "N"],
            [10, 10, 20, 20, 1.0, "B", "M"],
        ]

        bboxes_gt = [
            [10, 10, 20, 20, 1, "A", "N"],
            [10, 10, 20, 20, 1, "B", "M"],
        ]

        mAP = map_score(bboxes_det, bboxes_gt, 0.5)
        ans = 1
        self.assertEqual(mAP, ans)

    def test_map_score_range_1(self):
        """Test average mAP score within specified range with perfect detection using a list of 7 values as input"""
        bboxes_det = [
            [10, 10, 20, 20, 1.0, "A", "N"],
            [10, 10, 20, 20, 1.0, "B", "M"],
        ]

        bboxes_gt = [
            [10, 10, 20, 20, 1, "A", "N"],
            [10, 10, 20, 20, 1, "B", "M"],
        ]

        mAP_range = map_score_range(bboxes_det, bboxes_gt, 0.5, 0.95, 0.05)
        ans = 1
        self.assertEqual(mAP_range, ans)


if __name__ == "__main__":
    unittest.main()
