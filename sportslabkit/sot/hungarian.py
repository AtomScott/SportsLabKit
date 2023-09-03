from __future__ import annotations

from typing import Any

from sportslabkit.sot.base import SingleObjectTracker


class HungarianTracker(SingleObjectTracker):
    def __init__(
        self,
        target,
        initial_frame,
        detection_model=None,
        image_model=None,
        motion_model=None,
        matching_fn=None,
    ):
        super().__init__(
            target,
            pre_init_args={
                "initial_frame": initial_frame,
                "detection_model": detection_model,
                "image_model": image_model,
                "motion_model": motion_model,
                "matching_fn": matching_fn,
            },
        )

    def pre_initialize(self, initial_frame, detection_model, image_model, motion_model, matching_fn):
        self.detections = []
        self.detection_model = detection_model
        self.image_model = image_model
        self.motion_model = motion_model
        self.matching_fn: Any = matching_fn

        self.target["feature"] = self.image_model.embed_detections([self.target], initial_frame)[0]
        if self.motion_model is not None:
            self.motion_model.update(self.target)

    def update(self, current_frame):
        # Extract the new detections from the current frame
        current_frame = current_frame[0]
        detections = self.detection_model(current_frame)

        # update the motion model with the new detections
        if self.motion_model is not None:
            predictions = self.motion_model(self.tracklet)
            self.tracklet.update_current_observation("box", predictions)

        # extract features from the detections
        detections = detections[0].to_list()

        if len(detections) > 0 and self.image_model is not None:
            embeds = self.image_model.embed_detections(detections, current_frame)
            for i, det in enumerate(detections):
                det.feature = embeds[i]

        match = self.matching_fn([self.tracklet], detections)

        if len(match) > 0:  # if there is a match
            _, det_idx = match[0][0], match[0][1]
            new_state = {
                "box": detections[det_idx].box,
                "score": detections[det_idx].score,
                "feature": detections[det_idx].feature,
            }
        else:  # if there is no match
            new_state = {
                "box": self.tracklet.box,
                "score": 0.5,
                "feature": self.tracklet.feature,
            }

        if self.motion_model is not None:
            # update the motion model with the new detections
            self.motion_model.update(new_state)

        return new_state

    @property
    def required_keys(self):
        return ["box", "score", "feature"]
