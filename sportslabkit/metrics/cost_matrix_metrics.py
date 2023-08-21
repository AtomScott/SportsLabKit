from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from scipy.spatial.distance import cdist

from sportslabkit.checks import _check_cost_matrix, _check_detections, _check_trackers
from sportslabkit.metrics.object_detection import iou_score
from sportslabkit.types.detection import Detection
from sportslabkit.types.tracklet import Tracklet


class BaseCostMatrixMetric(ABC):
    """A base class for computing the cost matrix between trackers and
    detections."""

    def __call__(self, trackers: Sequence[Tracklet], detections: Sequence[Detection]) -> np.ndarray:
        """Calculate the metric between trackers and detections.

        Args:
            trackers: A list of trackers.
            detections: A list of detections.

        returns:
            An array of containing the metric between trackers and detections.
        """
        _check_trackers(trackers)
        _check_detections(detections)

        cost_matrix = self.compute_metric(trackers, detections)
        _check_cost_matrix(cost_matrix, trackers, detections)
        return cost_matrix

    @abstractmethod
    def compute_metric(self, trackers: Sequence[Tracklet], detections: Sequence[Detection]) -> np.ndarray:
        """Calculate the metric between trackers and detections.

        Args:
            trackers: A list of trackers.
            detections: A list of detections.

        returns:
            An array of containing the metric between trackers and detections.
        """
        raise NotImplementedError


class IoUCMM(BaseCostMatrixMetric):
    """Compute the IoU Cost Matrix Metric between trackers and detections."""

    def __init__(self, use_pred_box=False):
        self.use_pred_box = use_pred_box

    def compute_metric(self, trackers: Sequence[Tracklet], detections: Sequence[Detection]) -> np.ndarray:
        if self.use_pred_box:
            bb1 = np.array([(t.pred_box[0], t.pred_box[1], t.pred_box[0] + t.pred_box[2], t.pred_box[1] + t.pred_box[3]) for t in trackers])
        else:
            bb1 = np.array([(t.box[0], t.box[1], t.box[0] + t.box[2], t.box[1] + t.box[3]) for t in trackers])
        bb2 = np.array([(d.box[0], d.box[1], d.box[0] + d.box[2], d.box[1] + d.box[3]) for d in detections])
        return 1 - cdist(bb1, bb2, iou_score)


class EuclideanCMM(BaseCostMatrixMetric):
    """Compute the Euclidean Cost Matrix Metric between trackers and
    detections."""

    def __init__(self, use_pred_box=False, im_shape: tuple[float, float] = (1080, 1920)):
        self.normalizer = np.sqrt(im_shape[0] ** 2 + im_shape[1] ** 2)
        self.use_pred_box = use_pred_box
        

    def compute_metric(self, trackers: Sequence[Tracklet], detections: Sequence[Detection]) -> np.ndarray:
        if self.use_pred_box:
            centers1 = np.array([(t.pred_box[0] + t.pred_box[2] / 2, t.pred_box[1] + t.pred_box[3] / 2) for t in trackers])
        else:
            centers1 = np.array([(t.box[0] + t.box[2] / 2, t.box[1] + t.box[3] / 2) for t in trackers])

        centers2 = np.array([(d.box[0] + d.box[2] / 2, d.box[1] + d.box[3] / 2) for d in detections])
        return cdist(centers1, centers2) / self.normalizer  # keep values in [0, 1]


# FIXME: 技術負債を返済しましょう
class EuclideanCMM2D(BaseCostMatrixMetric):
    def __init__(self, im_shape: tuple[float, float] = (68, 105)):
        self.normalizer = np.sqrt(im_shape[0] ** 2 + im_shape[1] ** 2)

    def compute_metric(self, trackers: Sequence[Tracklet], detections: Sequence[Detection]) -> np.ndarray:
        t = trackers[0]
        d = detections[0]
        centers1 = np.array([(t.get_state("pitch_coordinates")[0], t.get_state("pitch_coordinates")[1]) for t in trackers])
        centers2 = np.array([(d.pitch_coordinates[0], d.pitch_coordinates[1]) for d in detections])
        return cdist(centers1, centers2) / self.normalizer  # keep values in [0, 1]


class CosineCMM(BaseCostMatrixMetric):
    """Compute the Cosine Cost Matrix Metric between trackers and
    detections."""

    def compute_metric(self, trackers: Sequence[Tracklet], detections: Sequence[Detection]) -> np.ndarray:
        vectors1 = np.array([t.feature for t in trackers])
        vectors2 = np.array([d.feature for d in detections])
        return cdist(vectors1, vectors2, "cosine") / 2  # keep values in [0, 1]
