import random
import numpy as np
import unittest

import soccertrack
from soccertrack.metrics import ap_score, iou_score, map_score, ap_score_range, map_score_range, mota_score, identity_score, hota_score


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
        
    def test_ap_score_range_1(self):
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
        
    def test_mota_score_1(self):
        """Test MOTA score with perfect detection."""
        dataset_path = soccertrack.datasets.get_path('top-view')
        path_to_csv = sorted(dataset_path.glob('annotations/*.csv'))[0]
        bbdf = soccertrack.load_df(path_to_csv)[0:2]
        bboxes_track = bbdf
        bboxes_gt = bbdf
        
        mota = mota_score(bboxes_track, bboxes_gt)
        ans = {'MOTA': 1.0,
            'MOTP': 1.0,
            'MODA': 1.0,
            'CLR_Re': 1.0,
            'CLR_Pr': 1.0,
            'MTR': 1.0,
            'PTR': 0.0,
            'MLR': 0.0,
            'sMOTA': 1.0,
            'CLR_F1': 1.0,
            'FP_per_frame': 0.0,
            'MOTAL': 1.0,
            'MOTP_sum': float(23 * len(bbdf)),
            'CLR_TP': 23*len(bbdf),
            'CLR_FN': 0,
            'CLR_FP': 0,
            'IDSW': 0,
            'MT': 23,
            'PT': 0,
            'ML': 0,
            'Frag': 0.0,
            'CLR_Frames': len(bbdf)}
        self.assertDictEqual(mota, ans)
        
        
    def identity_score_1(self):
        """Test IDENTITY score with perfect detection."""
        dataset_path = soccertrack.datasets.get_path('top-view')
        path_to_csv = sorted(dataset_path.glob('annotations/*.csv'))[0]
        bbdf = soccertrack.load_df(path_to_csv)[0:2]
        bboxes_track = bbdf
        bboxes_gt = bbdf
        
        identity = mota_score(bboxes_track, bboxes_gt)
        ans = {'IDF1': 1.0, 
            'IDR': 1.0, 
            'IDP': 1.0, 
            'IDTP': 23 * len(bbdf), 
            'IDFN': 0, 
            'IDFP': 0}

        self.assertDictEqual(identity, ans)
        
    def hota_score_1(self):
        """Test HOTA score with perfect detection."""
        dataset_path = soccertrack.datasets.get_path('top-view')
        path_to_csv = sorted(dataset_path.glob('annotations/*.csv'))[0]
        bbdf = soccertrack.load_df(path_to_csv)[0:2]
        bboxes_track = bbdf
        bboxes_gt = bbdf
        
        hota = mota_score(bboxes_track, bboxes_gt)
        ans = {'HOTA': 1.0,
            'DetA': 1.0,
            'AssA': 1.0,
            'DetRe': 1.0,
            'DetPr': 1.0,
            'AssRe': 1.0,
            'AssPr': 1.0,
            'LocA': 1.0,
            'RHOTA': 1.0,
            'HOTA_TP': float(23 * len(bbdf)),
            'HOTA_FN': 0.0,
            'HOTA_FP': 0.0,
            'HOTA(0)': 1.0,
            'LocA(0)': 1.0,
            'HOTALocA(0)': 1.0}

        self.assertDictEqual(hota, ans)



if __name__ == "__main__":
    unittest.main()
