from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from scipy.spatial.distance import cdist

from soccertrack.checks import _check_cost_matrix, _check_detections, _check_trackers
from soccertrack.metrics.object_detection import iou_score
from soccertrack.types import Detection, Tracker


class BaseCostMatrixMetric(ABC):
    """A base class for computing the cost matrix between trackers and
    detections."""

    def __call__(
        self, trackers: Sequence[Tracker], detections: Sequence[Detection]
    ) -> np.ndarray:
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
    def compute_metric(
        self, trackers: Sequence[Tracker], detections: Sequence[Detection]
    ) -> np.ndarray:
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

    def compute_metric(
        self, trackers: Sequence[Tracker], detections: Sequence[Detection]
    ) -> np.ndarray:
        bb1 = np.array(
            [
                (t.box[0], t.box[1], t.box[0] + t.box[2], t.box[1] + t.box[3])
                for t in trackers
            ]
        )
        bb2 = np.array(
            [
                (d.box[0], d.box[1], d.box[0] + d.box[2], d.box[1] + d.box[3])
                for d in detections
            ]
        )
        return 1 - cdist(bb1, bb2, iou_score)


class EuclideanCMM(BaseCostMatrixMetric):
    """Compute the Euclidean Cost Matrix Metric between trackers and
    detections."""

    def __init__(self, im_shape: tuple[float, float] = (1080, 1920)):
        self.normalizer = np.sqrt(im_shape[0] ** 2 + im_shape[1] ** 2)

    def compute_metric(
        self, trackers: Sequence[Tracker], detections: Sequence[Detection]
    ) -> np.ndarray:

        centers1 = np.array(
            [(t.box[0] + t.box[2] / 2, t.box[1] + t.box[3] / 2) for t in trackers]
        )
        centers2 = np.array(
            [(d.box[0] + d.box[2] / 2, d.box[1] + d.box[3] / 2) for d in detections]
        )

        return cdist(centers1, centers2) / self.normalizer  # keep values in [0, 1]


class CosineCMM(BaseCostMatrixMetric):
    """Compute the Cosine Cost Matrix Metric between trackers and
    detections."""

    def compute_metric(
        self, trackers: Sequence[Tracker], detections: Sequence[Detection]
    ) -> np.ndarray:
        vectors1 = np.array([t.feature for t in trackers])
        vectors2 = np.array([d.feature for d in detections])
        return cdist(vectors1, vectors2, "cosine") / 2  # keep values in [0, 1]
