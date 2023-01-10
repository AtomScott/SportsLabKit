""" assignment cost calculation & matching methods """

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import scipy
from scipy.spatial.distance import cdist

EPS = 1e-7


def calculate_iou(bboxes1, bboxes2, dim: int = 2):
    """expected bboxes size: (-1, 2*dim)"""
    bboxes1 = np.array(bboxes1).reshape((-1, dim * 2))
    bboxes2 = np.array(bboxes2).reshape((-1, dim * 2))

    coords_b1 = np.split(bboxes1, 2 * dim, axis=1)
    coords_b2 = np.split(bboxes2, 2 * dim, axis=1)

    coords = np.zeros(shape=(2, dim, bboxes1.shape[0], bboxes2.shape[0]))
    val_inter, val_b1, val_b2 = 1.0, 1.0, 1.0
    for d in range(dim):
        coords[0, d] = np.maximum(coords_b1[d], np.transpose(coords_b2[d]))  # top-left
        coords[1, d] = np.minimum(
            coords_b1[d + dim], np.transpose(coords_b2[d + dim])
        )  # bottom-right

        val_inter *= np.maximum(coords[1, d] - coords[0, d], 0)
        val_b1 *= coords_b1[d + dim] - coords_b1[d]
        val_b2 *= coords_b2[d + dim] - coords_b2[d]

    iou = val_inter / (
        np.clip(val_b1 + np.transpose(val_b2) - val_inter, a_min=0, a_max=None) + EPS
    )
    return iou


def angular_similarity(vectors1, vectors2):
    sim = 1 - cdist(vectors1, vectors2, "cosine") / 2  # kept in range <0,1>
    return sim


def _sequence_has_none(seq: Sequence[Any]) -> bool:
    return any([r is None for r in seq])


def cost_matrix_iou_feature(
    trackers: Sequence[Any],
    detections: Sequence[Any],
    feature_similarity_fn=angular_similarity,
    feature_similarity_beta: float = None,
) -> Tuple[np.ndarray, np.ndarray]:

    # boxes
    b1 = np.array([t.box() for t in trackers])
    b2 = np.array([d.box for d in detections])

    # box iou
    inferred_dim = int(len(b1[0]) / 2)
    iou_mat = calculate_iou(b1, b2, dim=inferred_dim)

    # feature similarity
    if feature_similarity_beta is not None:
        # get features
        f1 = [t.feature for t in trackers]
        f2 = [d.feature for d in detections]

        if _sequence_has_none(f1) or _sequence_has_none(f2):
            # fallback to pure IOU due to missing features
            apt_mat = iou_mat
        else:
            sim_mat = feature_similarity_fn(f1, f2)
            sim_mat = feature_similarity_beta + (1 - feature_similarity_beta) * sim_mat

            # combined aptitude
            apt_mat = np.multiply(iou_mat, sim_mat)
    else:
        apt_mat = iou_mat

    cost_mat = -1.0 * apt_mat
    return cost_mat, iou_mat


def cost_matrix_distance(trackers, detections, dim: int = 2):
    if len(trackers) == 0 or len(detections) == 0:
        return np.array([])

    bboxes1 = np.array([t.box() for t in trackers])
    bboxes2 = np.array([d.box for d in detections])

    centers1 = np.array([b.reshape(2, 2).mean(0) for b in bboxes1])
    centers2 = np.array([b.reshape(2, 2).mean(0) for b in bboxes2])

    return cdist(centers1, centers2)


def match_by_cost_matrix(
    cost_mat,
    max_cost: float = 0.1,
) -> np.ndarray:
    if len(cost_mat) == 0:
        return []

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_mat)

    matches = []
    for r, c in zip(row_ind, col_ind):
        # check linear assignment winner
        if cost_mat[r, c] <= max_cost:
            matches.append((r, c))

    return np.array(matches)


def match_by_affinity_matrix(
    affinity_mat,
    min_affinity: float = 0.1,
) -> np.ndarray:
    if len(affinity_mat) == 0:
        return []

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-affinity_mat)

    matches = []
    for r, c in zip(row_ind, col_ind):
        # check linear assignment winner
        if affinity_mat[r, c] >= min_affinity:
            matches.append((r, c))

    return np.array(matches)


class BaseMatchingFunction(ABC):
    """
    A base class for matching functions.
    """

    @abstractmethod
    def __call__(
        self, trackers: Sequence[Any], detections: Sequence[Any]
    ) -> np.ndarray:
        """Calculate the matching cost between trackers and detections.

        Args:
            trackers: A list of trackers.
            detections: A list of detections.

        returns:
            An array of containing indices of matching pairs of trackers and detections.
        """
        raise NotImplementedError()


class IOUAndFeatureMatchingFunction(BaseMatchingFunction):
    """class implements the basic matching function, taking into account
    detection boxes overlap measured using IOU metric and optional
    feature similarity measured with a specified metric"""

    def __init__(
        self,
        min_iou: float = 0.1,
        multi_match_min_iou: float = 1.0 + EPS,
        feature_similarity_fn: Callable = angular_similarity,
        feature_similarity_beta: Optional[float] = None,
    ) -> None:
        self.min_iou = min_iou
        self.multi_match_min_iou = multi_match_min_iou
        self.feature_similarity_fn = feature_similarity_fn
        self.feature_similarity_beta = feature_similarity_beta

    def __call__(
        self, trackers: Sequence[Any], detections: Sequence[Any]
    ) -> np.ndarray:
        return match_by_cost_matrix(
            trackers,
            detections,
            min_iou=self.min_iou,
            multi_match_min_iou=self.multi_match_min_iou,
            feature_similarity_fn=self.feature_similarity_fn,
            feature_similarity_beta=self.feature_similarity_beta,
        )


class EuclideanDistance(BaseMatchingFunction):
    def __init__(
        self,
        max_cost: float = 1000.0,
        conf_threshold_1: float = 0.5,
        conf_threshold_2: float = 0.01,
        multi_match_max_dist: float = 1.0 + EPS,
    ) -> None:
        self.max_cost = max_cost
        self.conf_threshold_1 = conf_threshold_1
        # self.conf_threshold_2

    def __call__(self, trackers, detections) -> np.ndarray:
        cost_mat = cost_matrix_distance(trackers, detections)
        return match_by_cost_matrix(
            cost_mat,
            max_cost=self.max_cost,
        )
