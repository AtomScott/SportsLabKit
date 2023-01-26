from __future__ import annotations

import collections
import dataclasses
import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

""" types """

# Box is of shape (1,2xdim), e.g. for dim=2 [xmin, ymin, width, height] format is accepted
Box = np.ndarray

# Vector is of shape (1, N)
Vector = np.ndarray

# Track is meant as an output from the object tracker
Track = collections.namedtuple("Track", "id box score class_id")

_pathlike = Union[str, Path]

# numpy/opencv image alias
NpImage = np.ndarray


class Tracker(ABC):
    pass


class Detection:
    """Detection.

    Note:
        BOX FORMAT: [xmin, ymin, width, height]
    """

    def __init__(
        self,
        box: Box,
        score: Optional[float] = None,
        class_id: Optional[int] = None,
        feature: Optional[Vector] = None,
    ):
        self.box = box
        self.score = score
        self.class_id = class_id
        self.feature = feature

    def __repr__(self):
        return f"Detection(box={self.box}, score={self.score:.5f}, class_id={self.class_id}, feature={self.feature})"
