# isort: skip_file

# Ordered for the documentation

from soccertrack.tracking_model.multi_object_tracker import MultiObjectTracker
from soccertrack.tracking_model.single_object_tracker import (
    SingleObjectTracker,
    KalmanTracker,
)
from soccertrack.tracking_model.matching import MotionVisualMatchingFunction

__all__ = [
    "MultiObjectTracker",
    "SingleObjectTracker",
    "KalmanTracker",
    "MotionVisualMatchingFunction",
]


# import numpy as np

# """ types """

# # Box is of shape (1,2xdim), e.g. for dim=2 [xmin, ymin, xmax, ymax] format is accepted
# Box = np.ndarray

# # Vector is of shape (1, N)
# Vector = np.ndarray

# # Track is meant as an output from the object tracker
# Track = collections.namedtuple("Track", "id box score class_id")


# # numpy/opencv image alias
# NpImage = np.ndarray


# class Detection:
#     def __init__(
#         self,
#         box: Box,
#         score: Optional[float] = None,
#         class_id: Optional[int] = None,
#         feature: Optional[Vector] = None,
#     ):
#         self.box = box
#         self.score = score
#         self.class_id = class_id
#         self.feature = feature

#     def __repr__(self):
#         return f"Detection(box={self.box}, score={self.score:.5f}, class_id={self.class_id}, feature={self.feature})"
