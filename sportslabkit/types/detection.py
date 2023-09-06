from __future__ import annotations

import numpy as np

from sportslabkit.types.types import Rect, Vector


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
        box: Rect | np.ndarray,
        score: float | None = None,
        class_id: int | None = None,
        feature: Vector | None = None,
    ):
        if isinstance(box, list):
            if len(box) != 4:
                raise ValueError(f"A list box should have exactly 4 elements, but got {len(box)}")
            box = Rect(*box)

        elif not isinstance(box, Rect):
            raise TypeError(f"Expected box to be of type Rect or List, got {type(box)}")

        self._box = box
        self._score = score
        self._class_id = class_id
        self._feature = feature

    @property
    def box(self) -> Rect:
        return self._box

    @box.setter
    def box(self, value: Rect):
        value = np.array(value).squeeze()
        if value.shape != (4,):
            raise ValueError(f"box should have the shape (4, ), but got {value.shape}")
        self._box = value

    @property
    def score(self) -> float | None:
        return self._score

    @score.setter
    def score(self, value: float | None):
        self._score = value

    @property
    def class_id(self) -> int | None:
        return self._class_id

    @class_id.setter
    def class_id(self, value: int | None):
        self._class_id = value

    @property
    def feature(self) -> Vector | None:
        return self._feature

    @feature.setter
    def feature(self, value: Vector | None):
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


# @dataclass
# class NEWDetection:
#     rect: Rect
#     class_id: int
#     class_name: str
#     confidence: float
#     tracker_id: int | None = None

#     @classmethod
#     def from_results(cls, pred: np.ndarray, names: dict[int, str]) -> list[NEWDetection]:
#         result = []
#         for x_min, y_min, x_max, y_max, confidence, class_id in pred:
#             class_id=int(class_id)
#             result.append(NEWDetection(
#                 rect=Rect(
#                     x=float(x_min),
#                     y=float(y_min),
#                     width=float(x_max - x_min),
#                     height=float(y_max - y_min)
#                 ),
#                 class_id=class_id,
#                 class_name=names[class_id],
#                 confidence=float(confidence)
#             ))
#         return result


# def filter_detections_by_class(detections: list[Detection], class_name: str) -> list[Detection]:
#     return [
#         detection
#         for detection
#         in detections
#         if detection.class_name == class_name
#     ]
