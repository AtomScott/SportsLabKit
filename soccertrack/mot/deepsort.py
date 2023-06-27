import soccertrack as st
from soccertrack.types import Tracklet
from soccertrack.mot.base import MultiObjectTracker
from soccertrack.matching import MotionVisualMatchingFunction
from soccertrack.motion_model import KalmanFilterMotionModel
from soccertrack.metrics import EuclideanCMM, CosineCMM
from soccertrack.logger import logger


class DeepSORTTracker(MultiObjectTracker):
    """DeepSORT tracker from https://arxiv.org/abs/1703.07402"""

    def __init__(
        self,
        detection_model=None,
        image_model=None,
        motion_model=None,
        motion_metric=EuclideanCMM(),
        motion_metric_beta=0.9,
        motion_metric_gate=0.2,
        visual_metric=CosineCMM(),
        visual_metric_beta=0.9,
        visual_metric_gate=0.2,
    ):
        super().__init__(
            pre_init_args={
                "detection_model": detection_model,
                "image_model": image_model,
                "motion_model": motion_model,
                "motion_metric": motion_metric,
                "motion_metric_beta": motion_metric_beta,
                "motion_metric_gate": motion_metric_gate,
                "visual_metric": visual_metric,
                "visual_metric_beta": visual_metric_beta,
                "visual_metric_gate": visual_metric_gate,
            },
        )

    def pre_initialize(
        self,
        detection_model,
        image_model,
        motion_model,
        motion_metric,
        motion_metric_beta,
        motion_metric_gate,
        visual_metric,
        visual_metric_beta,
        visual_metric_gate,
    ):
        if detection_model is None:
            # use yolov8 as default
            detection_model = st.detection_model.load("yolov8x")
        self.detection_model = detection_model

        if image_model is None:
            # use osnet as default
            image_model = st.image_model.load("osnet_x1_0")
        self.image_model = image_model

        if motion_model is None:
            motion_model = KalmanFilterMotionModel(dt=1 / 30, process_noise=0.1, measurement_noise=0.1)
        self.motion_model = motion_model

        self.matching_fn = MotionVisualMatchingFunction(
            motion_metric=motion_metric,
            motion_metric_beta=motion_metric_beta,
            motion_metric_gate=motion_metric_gate,
            visual_metric=visual_metric,
            visual_metric_beta=visual_metric_beta,
            visual_metric_gate=visual_metric_gate,
        )

    def update(self, current_frame, tracklets):
        # detect objects using the detection model
        detections = self.detection_model(current_frame)

        # update the motion model with the new detections
        # self.update_tracklets_with_motion_model_predictions
        for i, tracklet in enumerate(tracklets):
            predictions = self.motion_model(tracklet)
            tracklets[i].update_current_observation("box", predictions)

        # extract features from the detections
        detections = detections[0].to_list()
        if len(detections) > 0:
            embeds = self.image_model.embed_detections(detections, current_frame)
            for i, det in enumerate(detections):
                det.feature = embeds[i]

        matches = self.matching_fn(tracklets, detections)

        assigned_tracklets = []
        new_tracklets = []
        unassigned_tracklets = []

        # assigned tracklets: update
        for match in matches:
            track_idx, det_idx = match[0], match[1]
            tracklet = tracklets[track_idx]

            new_state = {
                "box": detections[det_idx].box,
                "score": detections[det_idx].score,
                "feature": detections[det_idx].feature,
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
                    "feature": det.feature,
                }
                new_tracklet = self.update_tracklet(Tracklet(), new_state)
                new_tracklets.append(new_tracklet)

        # unassigned tracklets: delete
        for i, tracklet in enumerate(tracklets):
            if i not in [match[0] for match in matches]:
                unassigned_tracklets.append(tracklet)

        logger.debug(
            f"assigned_tracklets: {len(assigned_tracklets)}, new_tracklets: {len(new_tracklets)}, unassigned_tracklets: {len(unassigned_tracklets)}"
        )
        return assigned_tracklets, new_tracklets, unassigned_tracklets

    @property
    def required_keys(self):
        return ["box", "score", "feature"]
