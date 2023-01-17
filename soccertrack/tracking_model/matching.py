"""assignment cost calculation & matching methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from soccertrack.checks import (
    _check_cost_matrix,
    _check_detections,
    _check_matches,
    _check_trackers,
)
from soccertrack.metrics import BaseCostMatrixMetric, CosineCMM, IoUCMM, iou_score
from soccertrack.tracking_model import SingleObjectTracker
from soccertrack.types import Detection

EPS = 1e-7


def linear_sum_assignment_with_inf(
    cost_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the linear sum assignment problem with inf values.

    Args:
        cost_matrix (np.ndarray): The cost matrix to solve.

    Raises:
        ValueError: Raises an error if the cost matrix contains both inf and -inf.
        ValueError: Raises an error if the cost matrix contains only inf or -inf.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The row and column indices of the assignment.
    """
    cost_matrix = np.asarray(cost_matrix)

    if cost_matrix.size == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int)

    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()

    if min_inf and max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if min_inf or max_inf:
        values = cost_matrix[~np.isinf(cost_matrix)]
        if values.size == 0:
            return np.empty((0,), dtype=int), np.empty((0,), dtype=int)

        m = values.min()
        M = values.max()
        n = min(cost_matrix.shape)
        positive = n * (M - m + np.abs(M) + np.abs(m) + 1)
        if max_inf:
            place_holder = (M + (n - 1) * (M - m)) + positive
        if min_inf:
            place_holder = (m + (n - 1) * (m - M)) - positive

        cost_matrix[np.isinf(cost_matrix)] = place_holder
    return linear_sum_assignment(cost_matrix)


class BaseMatchingFunction(ABC):
    """A base class for matching functions.

    A matching function takes a list of trackers and a list of
    detections and returns a list of matches. Subclasses should
    implement the :meth:`compute_cost_matrix` method.
    """

    def __call__(
        self, trackers: Sequence[SingleObjectTracker], detections: Sequence[Detection]
    ) -> np.ndarray:
        """Calculate the matching cost between trackers and detections.

        Args:
            trackers: A list of trackers.
            detections: A list of detections.

        returns:
            An array of containing indices of matching pairs of trackers and detections.
        """
        _check_trackers(trackers)
        _check_detections(detections)

        cost_matrix = self.compute_cost_matrix(trackers, detections)
        _check_cost_matrix(cost_matrix, trackers, detections)

        matches = self.match_cost_matrix(cost_matrix)
        _check_matches(matches, trackers, detections)

        return matches

    @abstractmethod
    def compute_cost_matrix(
        self, trackers: Sequence[SingleObjectTracker], detections: Sequence[Detection]
    ) -> np.ndarray:
        """Calculate the matching cost between trackers and detections.

        Args:
            trackers: A list of trackers.
            detections: A list of detections.

        returns:
            A 2D numpy array of matching costs between trackers and detections.
        """
        pass

    def match_cost_matrix(self, cost_matrix: np.ndarray) -> np.ndarray:
        """Match trackers and detections based on a cost matrix.

        While this method implements a hungarian algorithm, it is can be
        overriden by subclasses that implement different matching strategies.
        Args:
            cost_matrix: A 2D numpy array of matching costs between trackers and detections.

        returns:
            A 2D numpy array of shape (n, 2) containing indices of matching pairs of trackers and detections.
        """
        matches = np.array(linear_sum_assignment_with_inf(cost_matrix)).T
        return matches


class MotionVisualMatchingFunction(BaseMatchingFunction):
    """A matching function that uses a combination of motion and visual
    metrics.

    Args:
        motion_metric: A motion metric. Defaults to `IoUCMM`.
        motion_metric_beta: The weight of the motion metric. Defaults to 1.
        motion_metric_gate: The gate of the motion metric, i.e. if the
            motion metric is larger than this value, the cost will be
            set to infinity. Defaults to `np.inf`.
        visual_metric: A visual metric. Defaults to `CosineCMM`.
        visual_metric_beta: The weight of the visual metric. Defaults to 1.
        visual_metric_gate: The gate of the visual metric, i.e. if the
            visual metric is larger than this value, the cost will be
            set to infinity. Defaults to `np.inf`.

    Note:
        To implement your own matching function, you can inherit from `BaseMatchingFunction`
        and override the :meth:`compute_cost_matrix` method.
    """

    def __init__(
        self,
        motion_metric: BaseCostMatrixMetric = IoUCMM(),
        motion_metric_beta: float = 1,
        motion_metric_gate: float = np.inf,
        visual_metric: BaseCostMatrixMetric = CosineCMM(),
        visual_metric_beta: float = 1,
        visual_metric_gate: float = np.inf,
    ) -> None:
        if not isinstance(motion_metric, BaseCostMatrixMetric):
            raise TypeError("motion_metric should be a BaseCostMatrixMetric")
        if not isinstance(visual_metric, BaseCostMatrixMetric):
            raise TypeError("visual_metric should be a BaseCostMatrixMetric")

        self.motion_metric = motion_metric
        self.motion_metric_beta = motion_metric_beta
        self.motion_metric_gate = motion_metric_gate
        self.visual_metric = visual_metric
        self.visual_metric_beta = visual_metric_beta
        self.visual_metric_gate = visual_metric_gate

    def compute_cost_matrix(
        self, trackers: Sequence[SingleObjectTracker], detections: Sequence[Detection]
    ) -> np.ndarray:
        if len(trackers) == 0 or len(detections) == 0:
            return np.array([])

        # Compute motion cost
        motion_cost = self.motion_metric_beta * self.motion_metric(trackers, detections)

        # Gate elements of motion cost matrix to infinity
        motion_cost[motion_cost > self.motion_metric_gate] = np.inf

        # Compute visual cost
        visual_cost = self.visual_metric_beta * self.visual_metric(trackers, detections)

        # Gate elements of visual cost matrix to infinity
        visual_cost[visual_cost > self.visual_metric_gate] = np.inf

        # Compute total cost
        cost_matrix = motion_cost + visual_cost
        return cost_matrix
