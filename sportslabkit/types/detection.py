from __future__ import annotations

from typing import Optional

import numpy as np

from sportslabkit.types.types import Box, Vector


class Detection:
    """Detection represents an object detected in an image.

    The Detection class expects the following inputs:

    box (np.ndarray): The bounding box of the detected object. The shape is (4,).
    score (float, optional): The confidence score of the detection.
    class_id (int, optional): The class of the detected object.
    feature (np.ndarray, optional): The feature vector of the detected object. The shape is (1, N).

    Args:
        box (np.ndarray): The bounding box of the detected object. Should be an array of shape (4,).
        score (float, optional): The confidence score of the detection.
        class_id (int, optional): The class of the detected object.
        feature (np.ndarray, optional): The feature vector of the detected object. Should be an array of shape (1, N).

    Attributes:
        _box (np.ndarray): The bounding box of the detected object.
        _score (float, optional): The confidence score of the detection.
        _class_id (int, optional): The class of the detected object.
        _feature (np.ndarray, optional): The feature vector of the detected object.

    Raises:
        ValueError: If the box does not have the shape (4,).
    """

    def __init__(
        self,
        box: Box,
        score: Optional[float] = None,
        class_id: Optional[int] = None,
        feature: Optional[Vector] = None,
    ):
        box = np.array(box).squeeze()
        if box.shape != (4,):
            raise ValueError(f"box should have the shape (4, ), but got {box.shape}")

        self._box = box
        self._score = score
        self._class_id = class_id
        self._feature = feature

    @property
    def box(self) -> Box:
        return self._box

    @box.setter
    def box(self, value: Box):
        value = np.array(value).squeeze()
        if value.shape != (4,):
            raise ValueError(f"box should have the shape (4, ), but got {value.shape}")
        self._box = value

    @property
    def score(self) -> Optional[float]:
        return self._score

    @score.setter
    def score(self, value: Optional[float]):
        self._score = value

    @property
    def class_id(self) -> Optional[int]:
        return self._class_id

    @class_id.setter
    def class_id(self, value: Optional[int]):
        self._class_id = value

    @property
    def feature(self) -> Optional[Vector]:
        return self._feature

    @feature.setter
    def feature(self, value: Optional[Vector]):
        self._feature = value

    def __repr__(self):
        feature_str = str(self._feature) if self._feature is None else f"array of shape {self._feature.shape}"
        return f"Detection(box={self._box}, score={self._score:.5f}, class_id={self._class_id}, feature={feature_str})"

    def __eq__(self, other):
        if not isinstance(other, Detection):
            return NotImplemented
        return (
            np.array_equal(self._box, other.box)
            and np.isclose(self._score, other.score, atol=1e-5)
            and self._class_id == other.class_id
            and np.array_equal(self._feature, other.feature)
        )
