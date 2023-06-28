import soccertrack as st
from soccertrack.logger import logger
from soccertrack.matching import SimpleMatchingFunction
from soccertrack.metrics import IoUCMM
from soccertrack.mot.base import MultiObjectTracker
from soccertrack.motion_model import KalmanFilterMotionModel
from soccertrack.types import Tracklet


class SORTTracker(MultiObjectTracker):
    """SORT tracker from https://arxiv.org/pdf/1602.00763.pdf"""

    def __init__(
        self,
        detection_model=None,
        motion_model=None,
        metric=IoUCMM(),
        metric_gate=1.0,
        t_lost=1,
    ):
        super().__init__(
            pre_init_args={
                "detection_model": detection_model,
                "motion_model": motion_model,
                "metric": metric,
                "metric_gate": metric_gate,
                "t_lost": t_lost,
            }
        )

    def pre_initialize(
        self,
        detection_model,
        motion_model,
        metric,
        metric_gate,
        t_lost,
    ):
        if detection_model is None:
            # use yolov8 as default
            detection_model = st.detection_model.load("yolov8x")
        self.detection_model = detection_model

        if motion_model is None:
            motion_model = KalmanFilterMotionModel(
                dt=1 / 30, process_noise=0.1, measurement_noise=0.1
            )
        self.motion_model = motion_model

        self.matching_fn = SimpleMatchingFunction(
            metric=metric,
            gate=metric_gate,
        )
        self.t_lost = t_lost

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
            logger.debug(
                f"track_idx: {track_idx}, det_idx: {det_idx}, cost: {cost_matrix[track_idx, det_idx]}"
            )
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

        # unassigned tracklets: delete if staleness > t_lost
        for i, tracklet in enumerate(tracklets):
            if i not in [match[0] for match in matches]:
                staleness = tracklet.get_state("staleness")
                if staleness is None:
                    staleness = 0
                if staleness > self.t_lost:
                    unassigned_tracklets.append(tracklet)
                else:
                    tracklet.update_state("staleness", staleness + 1)
                    assigned_tracklets.append(tracklet)

        return assigned_tracklets, new_tracklets, unassigned_tracklets

    @property
    def required_observation_types(self):
        return ["box", "frame", "score"]

    @property
    def required_state_types(self):
        motion_model_required_state_types = self.motion_model.required_state_types
        required_state_types = motion_model_required_state_types + ["staleness"]
        return required_state_types
