from typing import Optional

import cv2 as cv
import numpy as np
from podm.bounding_box import BoundingBox
from podm.utils.enumerators import BBFormat

from soccertrack.utils import MovieIterator
from soccertrack.utils.camera import Camera


class CandidateDetection(BoundingBox):
    def __init__(
        self,
        camera: Camera,
        frame_idx: Optional[int] = None,
        detection_id: Optional[int] = None,
        detection_confidence: Optional[float] = None,
        from_bbox: Optional[BoundingBox] = None,
    ):
        self.camera = camera
        self.frame_idx = frame_idx
        if from_bbox is not None:
            bbox = from_bbox
            super().__init__(
                image_name=bbox._image_name,
                class_id=bbox._class_id,
                coordinates=bbox.get_absolute_bounding_box(),
                img_size=bbox.get_image_size(),
                bb_type=bbox._bb_type,
                confidence=bbox._confidence,
                # format=BBFormat.XYWH, # defualt should work
                # type_coordinates=CoordinatesType.ABSOLUTE, # default should work
            )
        else:
            raise NotImplementedError()

        self.px, self.py = camera.video2pitch(np.array([self._y, self._x])).squeeze()
        self.in_range = all(
            [
                self.camera.x_range[0] <= self.px <= self.camera.x_range[-1],
                self.camera.y_range[0] <= self.py <= self.camera.y_range[-1],
            ]
        )
