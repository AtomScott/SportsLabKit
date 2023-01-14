"""Checks for input arguments and outputs of functions."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from soccertrack.types import Detection, Tracker


def _check_trackers(trackers: Sequence[Tracker]) -> None:
    if not isinstance(trackers, Sequence):
        raise TypeError(
            f"trackers should be a sequence, but is {type(trackers).__name__}"
        )
    if not all(isinstance(t, Tracker) for t in trackers):
        raise TypeError(
            f"trackers should be a sequence of SingleObjectTracker, but "
            f"contains {type(trackers[0]).__name__}."
        )


def _check_detections(detections: Sequence[Detection]) -> None:
    if not isinstance(detections, Sequence):
        raise TypeError(
            f"detections should be a sequence, but is {type(detections).__name__}."
        )
    if not all(isinstance(d, Detection) for d in detections):
        raise TypeError(
            f"detections should be a sequence of Detection, but "
            f"contains {type(detections[0]).__name__}."
        )


def _check_cost_matrix(cost_matrix: np.ndarray, trackers, detections) -> None:
    if not isinstance(cost_matrix, np.ndarray):
        raise TypeError(
            f"cost_matrix should be a numpy array, but is {type(cost_matrix).__name__}."
        )
    if len(cost_matrix.shape) != 2:
        raise ValueError(
            f"cost_matrix should be a 2D array, but is {len(cost_matrix.shape)}D."
        )
    if cost_matrix.shape[0] != len(trackers):
        raise ValueError(
            f"cost_matrix should have {len(trackers)} rows, but has {cost_matrix.shape[0]}."
        )
    if cost_matrix.shape[1] != len(detections):
        raise ValueError(
            f"cost_matrix should have {len(detections)} columns, but has {cost_matrix.shape[1]}."
        )


def _check_matches(
    matches: np.ndarray,
    trackers: Sequence[Tracker],
    detections: Sequence[Detection],
) -> None:
    if not isinstance(matches, np.ndarray):
        raise TypeError(
            f"matches should be a numpy array, but is {type(matches).__name__}."
        )
    if len(matches.shape) != 2:
        raise ValueError(f"matches should be a 2D array, but is {len(matches.shape)}D.")
    if matches.shape[1] != 2:
        raise ValueError(f"matches should have 2 columns, but has {matches.shape[1]}.")
    if np.any(matches[:, 0] >= len(trackers)):
        raise ValueError(
            f"matches contains rows with tracker index greater than number of trackers."
        )
    if np.any(matches[:, 1] >= len(detections)):
        raise ValueError(
            f"matches contains rows with detection index greater than number of detections."
        )
