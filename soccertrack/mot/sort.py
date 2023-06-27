import soccertrack as st
from soccertrack.types import Tracklet
from soccertrack.mot.base import MultiObjectTracker
from soccertrack.matching import SimpleMatchingFunction
from soccertrack.motion_model import KalmanFilterMotionModel
from soccertrack.metrics import IoUCMM
from soccertrack.logger import logger


class SORTTracker(MultiObjectTracker):
    """SORT tracker from https://arxiv.org/pdf/1602.00763.pdf"""

    def __init__(
        self,
        detection_model=None,
        motion_model=None,
        metric=IoUCMM(),
        metric_gate=1.0,
    ):
        super().__init__(
            pre_init_args={
                "detection_model": detection_model,
                "motion_model": motion_model,
                "metric": metric,
                "metric_gate": metric_gate,
            },
        )

    def pre_initialize(
        self,
        detection_model,
        motion_model,
        metric,
        metric_gate,
    ):
        if detection_model is None:
            # use yolov8 as default
            detection_model = st.detection_model.load("yolov8x")
        self.detection_model = detection_model

        if motion_model is None:
            motion_model = KalmanFilterMotionModel(dt=1 / 30, process_noise=0.1, measurement_noise=0.1)
        self.motion_model = motion_model

        self.matching_fn = SimpleMatchingFunction(
            metric=metric,
            gate=metric_gate,
        )

    def update(self, current_frame, tracklets):
        # detect objects using the detection model
        detections = self.detection_model(current_frame)

        # update the motion model with the new detections
        # self.update_tracklets_with_motion_model_predictions
        current_boxes = []
        for i, tracklet in enumerate(tracklets):
            predictions = self.motion_model(tracklet)

            # FIXME : should overwrite the next observation not the current
            current_box = tracklet.get_observation("box")
            current_boxes.append(current_box)
            tracklet.update_current_observation("box", predictions)

        # extract features from the detections
        detections = detections[0].to_list()

        # Use predicted tracklets to match with detections since the order is the same
        matches, cost_matrix = self.matching_fn(tracklets, detections, True)

        for i, tracklet in enumerate(tracklets):
            tracklet.update_current_observation("box", current_boxes[i])

        assigned_tracklets = []
        new_tracklets = []
        unassigned_tracklets = []

        # assigned tracklets: update
        for match in matches:
            track_idx, det_idx = match[0], match[1]
            logger.debug(f"track_idx: {track_idx}, det_idx: {det_idx}, cost: {cost_matrix[track_idx, det_idx]}")
            tracklet = tracklets[track_idx]

            new_state = {
                "box": detections[det_idx].box,
                "score": detections[det_idx].score,
                "frame": self.frame_count,
            }

            # update the tracklet with the new state
            tracklet = self.update_tracklet(tracklet, new_state)
            assigned_tracklets.append(tracklet)

        # not assigned detections: create new trackers
        for i, det in enumerate(detections):
            if i not in [match[1] for match in matches]:
                new_state = {
                    "box": det.box,
                    "score": det.score,
                    "frame": self.frame_count,
                }
                new_tracklet = self.create_tracklet(new_state)
                new_tracklets.append(new_tracklet)

        # unassigned tracklets: delete
        for i, tracklet in enumerate(tracklets):
            if i not in [match[0] for match in matches]:
                unassigned_tracklets.append(tracklet)

        # print("assigned_tracklets")
        # for tracklet in assigned_tracklets:
        #     print(tracklet.id, tracklet.get_observation("box"))
        # print("new_tracklets")
        # for tracklet in new_tracklets:
        #     print(tracklet.id, tracklet.get_observation("box"))
        return assigned_tracklets, new_tracklets, unassigned_tracklets

    @property
    def required_observation_types(self):
        return ["box", "frame", "score"]

    @property
    def required_state_types(self):
        return self.motion_model.required_state_types
