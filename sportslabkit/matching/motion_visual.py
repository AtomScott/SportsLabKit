from __future__ import annotations

from typing import Sequence

import numpy as np

from sportslabkit import Tracklet
from sportslabkit.matching.base import BaseMatchingFunction
from sportslabkit.metrics import BaseCostMatrixMetric, CosineCMM, IoUCMM
from sportslabkit.types.detection import Detection


class MotionVisualMatchingFunction(BaseMatchingFunction):
    """A matching function that uses a combination of motion and visual
    metrics.

    Args:
        motion_metric: A motion metric. Defaults to `IoUCMM`.
        beta: The weight of the motion metric. The weight of the visual metric is calculated as 1 - beta. Defaults to 0.5.
        motion_metric_gate: The gate of the motion metric, i.e. if the
            motion metric is larger than this value, the cost will be
            set to infinity. Defaults to `np.inf`.
        visual_metric: A visual metric. Defaults to `CosineCMM`.
        visual_metric_gate: The gate of the visual metric, i.e. if the
            visual metric is larger than this value, the cost will be
            set to infinity. Defaults to `np.inf`.

    Note:
        To implement your own matching function, you can inherit from `BaseMatchingFunction`
        and override the :meth:`compute_cost_matrix` method.
    """

    hparam_search_space = {
        "beta": {"type": "float", "low": 0, "high": 1},
        "motion_metric_gate": {"type": "logfloat", "low": 1e-3, "high": 1e2},
        "visual_metric_gate": {"type": "logfloat", "low": 1e-3, "high": 1e2},
    }

    def __init__(
        self,
        motion_metric: BaseCostMatrixMetric = IoUCMM(),
        beta: float = 0.5,
        motion_metric_gate: float = np.inf,
        visual_metric: BaseCostMatrixMetric = CosineCMM(),
        visual_metric_gate: float = np.inf,
    ) -> None:
        if not isinstance(motion_metric, BaseCostMatrixMetric):
            raise TypeError("motion_metric should be a BaseCostMatrixMetric")
        if not isinstance(visual_metric, BaseCostMatrixMetric):
            raise TypeError("visual_metric should be a BaseCostMatrixMetric")

        self.motion_metric = motion_metric
        self.motion_metric_gate = motion_metric_gate
        self.visual_metric = visual_metric
        self.visual_metric_gate = visual_metric_gate
        self.beta = beta

    def compute_cost_matrix(self, trackers: Sequence[Tracklet], detections: Sequence[Detection]) -> np.ndarray:
        motion_metric_beta = self.beta
        visual_metric_beta = 1 - self.beta

        if len(trackers) == 0 or len(detections) == 0:
            return np.array([])

        # Compute motion cost
        motion_cost = motion_metric_beta * self.motion_metric(trackers, detections)

        # Gate elements of motion cost matrix to infinity
        motion_cost[motion_cost > self.motion_metric_gate] = np.inf

        # Compute visual cost
        visual_cost = visual_metric_beta * self.visual_metric(trackers, detections)

        # Gate elements of visual cost matrix to infinity
        visual_cost[visual_cost > self.visual_metric_gate] = np.inf

        # Compute total cost
        cost_matrix = motion_cost + visual_cost
        return cost_matrix
