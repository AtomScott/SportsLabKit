"""Detection utilities."""
from typing import Optional

import numpy as np
from podm.bounding_box import BoundingBox

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
        """Creates a new candidate detection.

        Args:
            camera (Camera): The camera that produced the detection.
            frame_idx (Optional[int], optional): Frame index. Defaults to None.
            detection_id (Optional[int], optional): Dection identification number Defaults to None.
            detection_confidence (Optional[float], optional): Confidence of bounding box. Defaults to None.
            from_bbox (Optional[BoundingBox], optional): `Bounding Box` to instantiate from. Defaults to None.

        Raises:
            NotImplementedError: Support for this instance without bbox is not implemented yet.

        Todo:
            * Implement support for this instance without bbox.
            * Add attribute documentation. Try to inherit from `BoundBox`.
            * Add examples.
        """
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
            super().__init__()
            self.deteection_id = detection_id
            self.detection_confidence = detection_confidence
            raise NotImplementedError()

        self.px, self.py = camera.video2pitch(np.array([self._y, self._x])).squeeze()
        self.in_range = all(
            [
                self.camera.x_range[0] <= self.px <= self.camera.x_range[-1],
                self.camera.y_range[0] <= self.py <= self.camera.y_range[-1],
            ]
        )
