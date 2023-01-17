import random
import unittest

import numpy as np
import pandas as pd

import soccertrack
from soccertrack.metrics import (
    ap_score,
    ap_score_range,
    hota_score,
    identity_score,
    iou_score,
    map_score,
    map_score_range,
    mota_score,
)

# global variables for tracking evaluation
dataset_path = soccertrack.datasets.get_path("top-view")
path_to_csv = sorted(dataset_path.glob("annotations/*.csv"))[0]
bbdf = soccertrack.load_df(path_to_csv)[0:2]
player_dfs = [player_df for _, player_df in bbdf.iter_players(drop=False)]


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
            "precision": [1.0, 1.0],
            "recall": [0.5, 1.0],
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
        """Test MOTA score with perfect tracking."""
        bboxes_track = bbdf
        bboxes_gt = bbdf

        mota = mota_score(bboxes_track, bboxes_gt)

        # create answer
        CLR_TP = 23 * 2
        CLR_FN = 0
        CLR_FP = 0
        IDSW = 0
        FP_per_frame = 0.0
        MT = 23
        PT = 0
        ML = 0
        Frag = 0
        MOTP_sum = float(23 * 2)  # Sum of similarity scores for matched bboxes

        MTR = MT / 23
        PTR = PT / 23
        MLR = ML / 23
        sMOTA = (MOTP_sum - CLR_FP - IDSW) / np.maximum(1.0, (CLR_TP + CLR_FN))
        MOTP = MOTP_sum / np.maximum(1.0, CLR_TP)
        CLR_Re = CLR_TP / np.maximum(1.0, (CLR_TP + CLR_FN))
        CLR_Pr = CLR_TP / np.maximum(1.0, (CLR_TP + CLR_FP))
        CLR_F1 = 2 * (CLR_Re * CLR_Pr) / np.maximum(1.0, (CLR_Re + CLR_Pr))
        MOTA = 1 - (CLR_FN + CLR_FP + IDSW) / np.maximum(1.0, (CLR_TP + CLR_FN))
        MODA = CLR_TP / np.maximum(1.0, (CLR_TP + CLR_FN))
        safe_log_idsw = np.log10(IDSW) if IDSW > 0 else IDSW
        MOTAL = (CLR_TP - CLR_FP - safe_log_idsw) / np.maximum(1.0, (CLR_TP + CLR_FN))

        ans = {
            "MOTA": MOTA,
            "MOTP": MOTP,
            "MODA": MODA,
            "CLR_Re": CLR_Re,
            "CLR_Pr": CLR_Pr,
            "MTR": MTR,
            "PTR": PTR,
            "MLR": MLR,
            "sMOTA": sMOTA,
            "CLR_F1": CLR_F1,
            "FP_per_frame": FP_per_frame,
            "MOTAL": MOTAL,
            "MOTP_sum": MOTP_sum,
            "CLR_TP": CLR_TP,
            "CLR_FN": CLR_FN,
            "CLR_FP": CLR_FP,
            "IDSW": IDSW,
            "MT": MT,
            "PT": PT,
            "ML": ML,
            "Frag": Frag,
            "CLR_Frames": len(bboxes_gt),
        }

        self.assertDictEqual(mota, ans)

    def test_mota_score_2(self):
        """Test MOTA score with zero tracking."""
        bboxes_track = bbdf[0:2].iloc[0:0]
        bboxes_gt = bbdf

        mota = mota_score(bboxes_track, bboxes_gt)

        # create answer
        CLR_TP = 0
        CLR_FN = 23 * 2
        CLR_FP = 0
        IDSW = 0
        FP_per_frame = 0.0
        MT = 0
        PT = 0
        ML = 23
        Frag = 0
        MOTP_sum = 0.0  # Sum of similarity scores for matched bboxes
        MTR = MT / 23
        PTR = PT / 23
        MLR = ML / 23
        sMOTA = (MOTP_sum - CLR_FP - IDSW) / np.maximum(1.0, (CLR_TP + CLR_FN))
        MOTP = MOTP_sum / np.maximum(1.0, CLR_TP)
        CLR_Re = CLR_TP / np.maximum(1.0, (CLR_TP + CLR_FN))
        CLR_Pr = CLR_TP / np.maximum(1.0, (CLR_TP + CLR_FP))
        CLR_F1 = 2 * (CLR_Re * CLR_Pr) / np.maximum(1.0, (CLR_Re + CLR_Pr))
        MOTA = 1 - (CLR_FN + CLR_FP + IDSW) / np.maximum(1.0, (CLR_TP + CLR_FN))
        MODA = CLR_TP / np.maximum(1.0, (CLR_TP + CLR_FN))
        safe_log_idsw = np.log10(IDSW) if IDSW > 0 else IDSW
        MOTAL = (CLR_TP - CLR_FP - safe_log_idsw) / np.maximum(1.0, (CLR_TP + CLR_FN))

        ans = {
            "MOTA": MOTA,
            "MOTP": MOTP,
            "MODA": MODA,
            "CLR_Re": CLR_Re,
            "CLR_Pr": CLR_Pr,
            "MTR": MTR,
            "PTR": PTR,
            "MLR": MLR,
            "sMOTA": sMOTA,
            "CLR_F1": CLR_F1,
            "FP_per_frame": FP_per_frame,
            "MOTAL": MOTAL,
            "MOTP_sum": MOTP_sum,
            "CLR_TP": CLR_TP,
            "CLR_FN": CLR_FN,
            "CLR_FP": CLR_FP,
            "IDSW": IDSW,
            "MT": MT,
            "PT": PT,
            "ML": ML,
            "Frag": Frag,
            "CLR_Frames": len(bboxes_gt),
        }

        self.assertDictEqual(mota, ans)

    def test_mota_score_3(self):
        """Test MOTA score with half tracking."""
        bboxes_track = player_dfs[0]
        bboxes_gt = pd.concat([player_dfs[0], player_dfs[1]], axis=1)

        mota = mota_score(bboxes_track, bboxes_gt)

        # create answer
        CLR_TP = 2
        CLR_FN = 2
        CLR_FP = 0
        IDSW = 0
        FP_per_frame = 0.0
        MT = 1
        PT = 0
        ML = 1
        Frag = 0
        MOTP_sum = 2.0  # Sum of similarity scores for matched bboxes
        MTR = MT / 2
        PTR = PT / 2
        MLR = ML / 2
        sMOTA = (MOTP_sum - CLR_FP - IDSW) / np.maximum(1.0, (CLR_TP + CLR_FN))
        MOTP = MOTP_sum / np.maximum(1.0, CLR_TP)
        CLR_Re = CLR_TP / np.maximum(1.0, (CLR_TP + CLR_FN))
        CLR_Pr = CLR_TP / np.maximum(1.0, (CLR_TP + CLR_FP))
        CLR_F1 = 2 * (CLR_Re * CLR_Pr) / np.maximum(1.0, (CLR_Re + CLR_Pr))
        MOTA = 1 - (CLR_FN + CLR_FP + IDSW) / np.maximum(1.0, (CLR_TP + CLR_FN))
        MODA = CLR_TP / np.maximum(1.0, (CLR_TP + CLR_FN))
        safe_log_idsw = np.log10(IDSW) if IDSW > 0 else IDSW
        MOTAL = (CLR_TP - CLR_FP - safe_log_idsw) / np.maximum(1.0, (CLR_TP + CLR_FN))

        ans = {
            "MOTA": MOTA,
            "MOTP": MOTP,
            "MODA": MODA,
            "CLR_Re": CLR_Re,
            "CLR_Pr": CLR_Pr,
            "MTR": MTR,
            "PTR": PTR,
            "MLR": MLR,
            "sMOTA": sMOTA,
            "CLR_F1": CLR_F1,
            "FP_per_frame": FP_per_frame,
            "MOTAL": MOTAL,
            "MOTP_sum": MOTP_sum,
            "CLR_TP": CLR_TP,
            "CLR_FN": CLR_FN,
            "CLR_FP": CLR_FP,
            "IDSW": IDSW,
            "MT": MT,
            "PT": PT,
            "ML": ML,
            "Frag": Frag,
            "CLR_Frames": len(bboxes_gt),
        }

        self.assertDictEqual(mota, ans)

    def test_mota_score_4(self):
        """Test for MOTA Score when an object is missing in the middle."""
        bboxes_track = player_dfs[0].copy()
        bboxes_track.loc[1] = -1
        bboxes_gt = pd.concat([player_dfs[0], player_dfs[1]], axis=1)

        mota = mota_score(bboxes_track, bboxes_gt)

        # create answer
        CLR_TP = 1
        CLR_FN = 3
        CLR_FP = 0
        IDSW = 0
        FP_per_frame = 0.0
        MT = 0
        PT = 1
        ML = 1
        Frag = 0
        MOTP_sum = 1.0  # Sum of similarity scores for matched bboxes
        MTR = MT / 2
        PTR = PT / 2
        MLR = ML / 2
        sMOTA = (MOTP_sum - CLR_FP - IDSW) / np.maximum(1.0, (CLR_TP + CLR_FN))
        MOTP = MOTP_sum / np.maximum(1.0, CLR_TP)
        CLR_Re = CLR_TP / np.maximum(1.0, (CLR_TP + CLR_FN))
        CLR_Pr = CLR_TP / np.maximum(1.0, (CLR_TP + CLR_FP))
        CLR_F1 = 2 * (CLR_Re * CLR_Pr) / np.maximum(1.0, (CLR_Re + CLR_Pr))
        MOTA = 1 - (CLR_FN + CLR_FP + IDSW) / np.maximum(1.0, (CLR_TP + CLR_FN))
        MODA = CLR_TP / np.maximum(1.0, (CLR_TP + CLR_FN))
        safe_log_idsw = np.log10(IDSW) if IDSW > 0 else IDSW
        MOTAL = (CLR_TP - CLR_FP - safe_log_idsw) / np.maximum(1.0, (CLR_TP + CLR_FN))

        ans = {
            "MOTA": MOTA,
            "MOTP": MOTP,
            "MODA": MODA,
            "CLR_Re": CLR_Re,
            "CLR_Pr": CLR_Pr,
            "MTR": MTR,
            "PTR": PTR,
            "MLR": MLR,
            "sMOTA": sMOTA,
            "CLR_F1": CLR_F1,
            "FP_per_frame": FP_per_frame,
            "MOTAL": MOTAL,
            "MOTP_sum": MOTP_sum,
            "CLR_TP": CLR_TP,
            "CLR_FN": CLR_FN,
            "CLR_FP": CLR_FP,
            "IDSW": IDSW,
            "MT": MT,
            "PT": PT,
            "ML": ML,
            "Frag": Frag,
            "CLR_Frames": len(bboxes_gt),
        }
        self.assertDictEqual(mota, ans)

    def test_identity_score_1(self):
        """Test IDENTITY score with perfect detection."""
        bboxes_track = bbdf
        bboxes_gt = bbdf

        identity = identity_score(bboxes_track, bboxes_gt)

        IDTP = 46
        IDFN = 0
        IDFP = 0
        IDR = IDTP / (IDTP + IDFN)
        IDP = IDTP / (IDTP + IDFP)
        IDF1 = 2 * (IDR * IDP) / (IDR + IDP)
        ans = {
            "IDF1": IDF1,
            "IDR": IDR,
            "IDP": IDP,
            "IDTP": IDTP,
            "IDFN": IDFN,
            "IDFP": IDFP,
        }
        self.assertDictEqual(identity, ans)

    def test_identity_score_2(self):
        """Test IDENTITY score with zero tracking."""
        bboxes_track = bbdf[0:2].iloc[0:0]
        bboxes_gt = bbdf

        identity = identity_score(bboxes_track, bboxes_gt)

        IDTP = 0
        IDFN = 46
        IDFP = 0
        IDR = IDTP / (IDTP + IDFN)
        IDP = 0
        IDF1 = 0
        ans = {
            "IDF1": IDF1,
            "IDR": IDR,
            "IDP": IDP,
            "IDTP": IDTP,
            "IDFN": IDFN,
            "IDFP": IDFP,
        }
        self.assertDictEqual(identity, ans)

    def test_identity_score_3(self):
        """Test IDENTITY score with half tracking."""
        bboxes_track = player_dfs[0]
        bboxes_gt = pd.concat([player_dfs[0], player_dfs[1]], axis=1)

        identity = identity_score(bboxes_track, bboxes_gt)

        IDTP = 2
        IDFN = 2
        IDFP = 0
        IDR = IDTP / (IDTP + IDFN)
        IDP = IDTP / (IDTP + IDFP)
        IDF1 = 2 * (IDR * IDP) / (IDR + IDP)
        ans = {
            "IDF1": IDF1,
            "IDR": IDR,
            "IDP": IDP,
            "IDTP": IDTP,
            "IDFN": IDFN,
            "IDFP": IDFP,
        }
        self.assertDictEqual(identity, ans)

    def test_identity_score_4(self):
        """Test for IDENTITY Score when an object is missing in the middle."""
        bboxes_track = player_dfs[0].copy()
        bboxes_track.loc[1] = -1
        bboxes_gt = pd.concat([player_dfs[0], player_dfs[1]], axis=1)

        identity = identity_score(bboxes_track, bboxes_gt)

        IDTP = 1
        IDFN = 3
        IDFP = 0
        IDR = IDTP / (IDTP + IDFN)
        IDP = IDTP / (IDTP + IDFP)
        IDF1 = 2 * (IDR * IDP) / (IDR + IDP)
        ans = {
            "IDF1": IDF1,
            "IDR": IDR,
            "IDP": IDP,
            "IDTP": IDTP,
            "IDFN": IDFN,
            "IDFP": IDFP,
        }
        self.assertDictEqual(identity, ans)

    def test_hota_score_1(self):
        """Test HOTA score with perfect detection."""
        bboxes_track = bbdf
        bboxes_gt = bbdf

        hota = hota_score(bboxes_track, bboxes_gt)

        # calculate answers
        HOTA_TP = 46.0
        HOTA_FN = 0.0
        HOTA_FP = 0.0

        AssA = 1.0
        AssRe = 1.0
        AssPr = 1.0
        LocA = 1.0
        DetA = HOTA_TP / np.maximum(1.0, (HOTA_TP + HOTA_FN + HOTA_FP))
        DetRe = HOTA_TP / np.maximum(1.0, (HOTA_TP + HOTA_FN))
        DetPr = HOTA_TP / np.maximum(1.0, (HOTA_TP + HOTA_FP))
        HOTA = np.sqrt(DetA * AssA)
        RHOTA = np.sqrt(DetA * AssA)
        HOTA0 = np.sqrt(DetA * AssA)
        LocA0 = 1.0
        HOTALocA0 = np.sqrt(DetA * AssA) * LocA0

        ans = {
            "HOTA": HOTA,
            "DetA": DetA,
            "AssA": AssA,
            "DetRe": DetRe,
            "DetPr": DetPr,
            "AssRe": AssRe,
            "AssPr": AssPr,
            "LocA": LocA,
            "RHOTA": RHOTA,
            "HOTA_TP": HOTA_TP,
            "HOTA_FN": HOTA_FN,
            "HOTA_FP": HOTA_FP,
            "HOTA(0)": HOTA0,
            "LocA(0)": LocA0,
            "HOTALocA(0)": HOTALocA0,
        }

        self.assertDictEqual(hota, ans)

    def test_hota_score_2(self):
        """Test HOTA score with zero detection."""
        bboxes_track = bbdf[0:2].iloc[0:0]
        bboxes_gt = bbdf

        hota = hota_score(bboxes_track, bboxes_gt)

        # calculate answers
        HOTA_TP = 0.0
        HOTA_FN = 46.0
        HOTA_FP = 0.0

        AssA = 0.0
        AssRe = 0.0
        AssPr = 0.0
        LocA = 1.0
        DetA = HOTA_TP / np.maximum(1.0, (HOTA_TP + HOTA_FN + HOTA_FP))
        DetRe = HOTA_TP / np.maximum(1.0, (HOTA_TP + HOTA_FN))
        DetPr = HOTA_TP / np.maximum(1.0, (HOTA_TP + HOTA_FP))
        HOTA = np.sqrt(DetA * AssA)
        RHOTA = np.sqrt(DetA * AssA)
        HOTA0 = np.sqrt(DetA * AssA)
        LocA0 = 1.0
        HOTALocA0 = np.sqrt(DetA * AssA) * LocA0

        ans = {
            "HOTA": HOTA,
            "DetA": DetA,
            "AssA": AssA,
            "DetRe": DetRe,
            "DetPr": DetPr,
            "AssRe": AssRe,
            "AssPr": AssPr,
            "LocA": LocA,
            "RHOTA": RHOTA,
            "HOTA_TP": HOTA_TP,
            "HOTA_FN": HOTA_FN,
            "HOTA_FP": HOTA_FP,
            "HOTA(0)": HOTA0,
            "LocA(0)": LocA0,
            "HOTALocA(0)": HOTALocA0,
        }

        self.assertDictEqual(hota, ans)

    def test_hota_score_3(self):
        """Test HOTA score with half detection."""
        bboxes_track = player_dfs[0]
        bboxes_gt = pd.concat([player_dfs[0], player_dfs[1]], axis=1)

        hota = hota_score(bboxes_track, bboxes_gt)

        # calculate answers
        HOTA_TP = 2.0
        HOTA_FN = 2.0
        HOTA_FP = 0.0

        AssA = 1.0
        AssRe = 1.0
        AssPr = 1.0
        LocA = 1.0
        DetA = HOTA_TP / (HOTA_TP + HOTA_FN + HOTA_FP)
        DetRe = HOTA_TP / (HOTA_TP + HOTA_FN)
        DetPr = HOTA_TP / (HOTA_TP + HOTA_FP)
        HOTA = np.sqrt(DetA * AssA)
        RHOTA = np.sqrt(DetA * AssA)
        HOTA0 = np.sqrt(DetA * AssA)
        LocA0 = 1.0
        HOTALocA0 = np.sqrt(DetA * AssA) * LocA0

        ans = {
            "HOTA": HOTA,
            "DetA": DetA,
            "AssA": AssA,
            "DetRe": DetRe,
            "DetPr": DetPr,
            "AssRe": AssRe,
            "AssPr": AssPr,
            "LocA": LocA,
            "RHOTA": RHOTA,
            "HOTA_TP": HOTA_TP,
            "HOTA_FN": HOTA_FN,
            "HOTA_FP": HOTA_FP,
            "HOTA(0)": HOTA0,
            "LocA(0)": LocA0,
            "HOTALocA(0)": HOTALocA0,
        }

        self.assertDictEqual(hota, ans)

    def test_hota_score_4(self):
        """Test for HOTA Score when an object is missing in the middle."""

        bboxes_track = player_dfs[0].copy()
        bboxes_track.loc[1] = -1
        bboxes_gt = pd.concat([player_dfs[0], player_dfs[1]], axis=1)

        hota = hota_score(bboxes_track, bboxes_gt)

        # calculate answers
        HOTA_TP = 1.0
        HOTA_FN = 3.0
        HOTA_FP = 0.0

        AssA = 1.0 / 2.0
        AssRe = 0.5
        AssPr = 1.0
        LocA = 1.0
        DetA = HOTA_TP / (HOTA_TP + HOTA_FN + HOTA_FP)
        DetRe = HOTA_TP / (HOTA_TP + HOTA_FN)
        DetPr = HOTA_TP / (HOTA_TP + HOTA_FP)
        HOTA = np.sqrt(DetA * AssA)
        RHOTA = np.sqrt(DetA * AssA)
        HOTA0 = np.sqrt(DetA * AssA)
        LocA0 = 1.0
        HOTALocA0 = np.sqrt(DetA * AssA) * LocA0

        ans = {
            "HOTA": HOTA,
            "DetA": DetA,
            "AssA": AssA,
            "DetRe": DetRe,
            "DetPr": DetPr,
            "AssRe": AssRe,
            "AssPr": AssPr,
            "LocA": LocA,
            "RHOTA": RHOTA,
            "HOTA_TP": HOTA_TP,
            "HOTA_FN": HOTA_FN,
            "HOTA_FP": HOTA_FP,
            "HOTA(0)": HOTA0,
            "LocA(0)": LocA0,
            "HOTALocA(0)": HOTALocA0,
        }

        self.assertDictEqual(hota, ans)


if __name__ == "__main__":
    unittest.main()
