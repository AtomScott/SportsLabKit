import os
import sys
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

import unittest

import numpy as np

from soccertrack.logger import *
from soccertrack.tracking_model import MotionVisualMatchingFunction, SingleObjectTracker
from soccertrack.types import Detection

dets = [
    Detection([10, 10, 5, 5], 0.9, 0, [1, 2, 3]),
    Detection([5, 10, 10, 10], 0.9, 0, [1, 1, 0]),
    Detection([15, 10, 10, 10], 0.9, 0, [1, 10, 0]),
]

_det0 = Detection([5, 5, 15, 15], 0.45, 0, [0, 3, 3])
_det1 = Detection([10, 10, 5, 5], 0.9, 0, [1, 2, 3])

trackers = []
for det in [_det0, _det1]:
    sot = SingleObjectTracker()
    sot.update(detection=det)
    trackers.append(sot)


class TestMatchingFunction(unittest.TestCase):
    def test_MotionVisualMatchingFunction_no_gates(self):
        matching_fn = MotionVisualMatchingFunction(
            motion_metric_gate=np.inf, visual_metric_gate=np.inf
        )
        matches = matching_fn(trackers=trackers, detections=dets)

        self.assertEqual(type(matches), np.ndarray)
        self.assertEqual(matches.shape, (2, 2))

        matches = matches.tolist()

        self.assertListEqual(matches[0], [0, 1])
        self.assertListEqual(matches[1], [1, 0])

        # # Calculate IOU cost matrix
        # bboxes1 = np.array([t.box for t in trackers])
        # bboxes2 = np.array([d.box for d in detections])
        # iou_matrix = 1 - cdist(bboxes1, bboxes2, calculate_iou)
        # if self.feature_similarity_beta is None:
        #     return iou_matrix

        # # Calculate feature cost matrix
        # features1 = np.array([t.feature for t in trackers])
        # features2 = np.array([d.feature for d in detections])
        # feature_matrix = cdist(features1, features2, self.feature_similarity_fn)

        # return iou_matrix + self.feature_similarity_beta * feature_matrix
