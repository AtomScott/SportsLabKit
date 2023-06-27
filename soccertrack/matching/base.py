"""assignment cost calculation & matching methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from soccertrack import Tracklet
from soccertrack.checks import (
    _check_cost_matrix,
    _check_detections,
    _check_matches,
    _check_trackers,
)
from soccertrack.metrics import BaseCostMatrixMetric, CosineCMM, IoUCMM
from soccertrack.types.detection import Detection

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

        # m = values.min()
        # M = values.max()
        # n = min(cost_matrix.shape)
        # positive = n * (M - m + np.abs(M) + np.abs(m) + 1)
        if max_inf:
            place_holder = np.finfo(
                cost_matrix.dtype
            ).max  # (M + (n - 1) * (M - m)) + positive
        if min_inf:
            place_holder = np.finfo(
                cost_matrix.dtype
            ).min  # (m + (n - 1) * (m - M)) - positive
        cost_matrix[np.isinf(cost_matrix)] = place_holder

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    if min_inf or max_inf:
        # Filter out matches with the placeholder value
        valid_indices = cost_matrix[row_ind, col_ind] != place_holder
        return row_ind[valid_indices], col_ind[valid_indices]
    return row_ind, col_ind


class BaseMatchingFunction(ABC):
    """A base class for matching functions.

    A matching function takes a list of trackers and a list of
    detections and returns a list of matches. Subclasses should
    implement the :meth:`compute_cost_matrix` method.
    """

    def __call__(
        self,
        trackers: Sequence[Tracklet],
        detections: Sequence[Detection],
        return_cost_matrix: bool = False,
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

        if return_cost_matrix:
            return matches, cost_matrix
        return matches

    @abstractmethod
    def compute_cost_matrix(
        self, trackers: Sequence[Tracklet], detections: Sequence[Detection]
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
