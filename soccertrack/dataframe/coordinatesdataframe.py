from __future__ import annotations

import json
from ast import literal_eval
from pathlib import Path
from typing import Mapping, Optional, Union
from xml.etree import ElementTree

import cv2
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .base import SoccerTrackMixin

_pathlike = Union[str, Path]


class CoordinatesDataFrame(SoccerTrackMixin, pd.DataFrame):

    _metadata = [
        "source_keypoints",
        "target_keypoints",
    ]

    @property
    def _constructor(self):
        return CoordinatesDataFrame

    @property
    def H(self) -> NDArray[np.float64]:
        """Calculate the homography transformation matrix from pitch to video
        space.

        Returns:
            NDArray[np.float64]: homography transformation matrix.
        """
        H, *_ = cv2.findHomography(
            self.source_keypoints, self.target_keypoints, cv2.RANSAC, 5.0
        )
        return H

    def set_keypoints(
        self,
        source_keypoints: Optional[ArrayLike] = None,
        target_keypoints: Optional[ArrayLike] = None,
        mapping: Optional[Mapping] = None,
        mapping_file: Optional[_pathlike] = None,
    ) -> None:
        """Set the keypoints for the homography transformation. Make sure that the target keypoints are the pitch coordinates. Also each keypoint must be a tuple of (Lon, Lat) or (x, y) coordinates.

        Args:
            source_keypoints (Optional[ArrayLike], optional): Keypoints in pitch space. Defaults to None.
            target_keypoints (Optional[ArrayLike], optional): Keypoints in video space. Defaults to None.
        """

        if mapping_file is not None:
            with open(mapping_file, "r") as f:
                mapping = json.load(f)
        if mapping is not None:
            target_keypoints, source_keypoints = [], []
            for target_kp, source_kp in mapping.items():
                if isinstance(target_kp, str):
                    target_kp = literal_eval(target_kp)
                if isinstance(source_kp, str):
                    source_kp = literal_eval(source_kp)
                target_keypoints.append(target_kp)
                source_keypoints.append(source_kp)

        self.source_keypoints = np.array(source_keypoints)
        self.target_keypoints = np.array(target_keypoints)

    def to_pitch_coordinates(self, drop=True):
        """Convert image coordinates to pitch coordinates.
        """
        transformed_groups = []
        for i, g in self.iter_players():
            pts = g[[(i[0], i[1], 'Lon'), (i[0], i[1], 'Lat')]].values
            x, y = cv2.perspectiveTransform(np.asarray([pts]), self.H).squeeze().T
            g[(i[0], i[1], 'x')] = x
            g[(i[0], i[1], 'y')] = y
            
            if drop:
                g.drop(columns=[(i[0], i[1], 'Lon'), (i[0], i[1], 'Lat')], inplace=True)
            transformed_groups.append(g)

        return self._constructor(pd.concat(transformed_groups, axis=1))
    
    # @property
    # def _constructor_sliced(self):
    #     raise NotImplementedError("This pandas method constructs pandas.Series object, which is not yet implemented in {self.__name__}.")
