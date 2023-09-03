import sportslabkit as st
from sportslabkit.logger import logger
from sportslabkit.matching import MotionVisualMatchingFunction
from sportslabkit.metrics import CosineCMM, IoUCMM
from sportslabkit.mot.base import MultiObjectTracker
from sportslabkit.motion_model import KalmanFilter


class DeepSORTTracker(MultiObjectTracker):
    """DeepSORT tracker from https://arxiv.org/abs/1703.07402"""

    hparam_search_space = {
        "max_staleness": {"type": "int", "low": 1, "high": 1e3},
        "min_length": {"type": "int", "low": 1, "high": 1e3},
    }

    def __init__(
        self,
        detection_model=None,
        image_model=None,
        motion_model=None,
        matching_fn: MotionVisualMatchingFunction=MotionVisualMatchingFunction(
            motion_metric=IoUCMM(),
            motion_metric_gate=0.2,
            visual_metric=CosineCMM(),
            visual_metric_gate=0.2,
            beta=0.5,
        ),
        window_size: int = 1,
        step_size: int = None,
        max_staleness: int = 5,
        min_length: int = 5,
    ):
        super().__init__(
            window_size=window_size,
            step_size=step_size,
            max_staleness=max_staleness,
            min_length=min_length,
        )
        self.detection_model = detection_model
        self.image_model = image_model
        self.motion_model = motion_model
        self.matching_fn = matching_fn

    def update(self, current_frame, tracklets):
        # detect objects using the detection model
        detections = self.detection_model(current_frame)

        # update the motion model with the new detections
        # self.update_tracklets_with_motion_model_predictions
        for i, tracklet in enumerate(tracklets):
            # `predicted_box` should be in form [bbleft, bbtop, bbwidth, bbheight]
            predicted_box = self.motion_model(tracklet)
            tracklet.update_state("pred_box", predicted_box)

        detections = detections[0].to_list()

        # extract features from the detections
        if len(detections) > 0:
            embeds = self.image_model.embed_detections(detections, current_frame)
            for i, det in enumerate(detections):
                det.feature = embeds[i]

        # Use predicted tracklets to match with detections since the order is the same
        matches, cost_matrix = self.matching_fn(tracklets, detections, return_cost_matrix=True)

        assigned_tracklets = []
        new_tracklets = []
        unassigned_tracklets = []

        # assigned tracklets: update
        for match in matches:
            track_idx, det_idx = match[0], match[1]
            tracklet = tracklets[track_idx]
            logger.debug(f"track_idx: {track_idx}, det_idx: {det_idx}, cost: {cost_matrix[track_idx, det_idx]}, track staleness: {tracklet.get_state('staleness')}")

            new_observation = {
                "box": detections[det_idx].box,
                "score": detections[det_idx].score,
                "frame": self.frame_count,
                "feature": detections[det_idx].feature,
            }

            # update the tracklet with the new state
            tracklet = self.update_tracklet(tracklet, new_observation)
            assigned_tracklets.append(tracklet)

        # not assigned detections: create new trackers
        for i, det in enumerate(detections):
            if i not in [match[1] for match in matches]:
                new_observation = {
                    "box": det.box,
                    "score": det.score,
                    "frame": self.frame_count,
                    "feature": det.feature,
                }
                new_tracklet = self.create_tracklet(new_observation)
                new_tracklets.append(new_tracklet)

        # unassigned tracklets: update tracklet with predicted state
        for i, tracklet in enumerate(tracklets):
            if i not in [match[0] for match in matches]:
                new_observation = {
                    "box": tracklet.get_state("pred_box"),
                    "score": tracklet.get_observation("score"),
                    "frame": self.frame_count,
                    "feature": tracklet.get_observation("feature"),
                }
                tracklet = self.update_tracklet(tracklet, new_observation)
                unassigned_tracklets.append(tracklet)

        return assigned_tracklets, new_tracklets, unassigned_tracklets

    @property
    def required_observation_types(self):
        return ["box", "score", "feature", "frame"]

    @property
    def required_state_types(self):
        motion_model_required_state_types = self.motion_model.required_state_types
        required_state_types = motion_model_required_state_types + ["pred_box"]
        return required_state_types
