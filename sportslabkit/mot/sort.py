from sportslabkit.logger import logger
from sportslabkit.matching import SimpleMatchingFunction
from sportslabkit.metrics import IoUCMM
from sportslabkit.mot.base import MultiObjectTracker
from typing import Optional


class SORTTracker(MultiObjectTracker):
    """SORT tracker from https://arxiv.org/pdf/1602.00763.pdf"""

    hparam_search_space = {
        "max_staleness": {"type": "int", "low": 1, "high": 1e3},
        "min_length": {"type": "int", "low": 1, "high": 1e3},
    }

    def __init__(
        self,
        detection_model,
        motion_model,
        matching_fn: SimpleMatchingFunction = SimpleMatchingFunction(metric = IoUCMM(use_pred_box=True), gate=1.0),
        window_size: int = 1,
        step_size: Optional[int] = None,
        max_staleness: int = 5,
        min_length: int = 5,
    ):
        """
        Initializes the SORT Tracker.

        Args:
            detection_model (Any): The model used for object detection.
            motion_model (Any): The model used for motion prediction.
            metric (IoUCMM, optional): The metric used for matching. Defaults to IoUCMM().
            metric_gate (float, optional): The gating threshold for the metric. Defaults to 1.0.
        """
        super().__init__(
            window_size=window_size,
            step_size=step_size,
            max_staleness=max_staleness,
            min_length=min_length,
        )

        self.detection_model = detection_model
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

        # Use predicted tracklets to match with detections since the order is the same
        matches, cost_matrix = self.matching_fn(tracklets, detections, return_cost_matrix=True)

        assigned_tracklets = []
        new_tracklets = []
        unassigned_tracklets = []

        # assigned tracklets (& detections): update tracklet with detection
        for match in matches:
            track_idx, det_idx = match[0], match[1]
            tracklet = tracklets[track_idx]
            logger.debug(f"track_idx: {track_idx}, det_idx: {det_idx}, cost: {cost_matrix[track_idx, det_idx]}, track staleness: {tracklet.get_state('staleness')}")

            new_observation = {
                "box": detections[det_idx].box,
                "score": detections[det_idx].score,
                "frame": self.frame_count,
            }

            # update the tracklet with the new state
            tracklet = self.update_tracklet(tracklet, new_observation)
            assigned_tracklets.append(tracklet)

        # unassigned detections: create new trackers
        for i, det in enumerate(detections):
            if i not in [match[1] for match in matches]:
                new_observation = {
                    "box": det.box,
                    "score": det.score,
                    "frame": self.frame_count,
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
                }
                tracklet = self.update_tracklet(tracklet, new_observation)
                unassigned_tracklets.append(tracklet)

        return assigned_tracklets, new_tracklets, unassigned_tracklets

    @property
    def required_observation_types(self):
        return ["box", "frame", "score"]

    @property
    def required_state_types(self):
        motion_model_required_state_types = self.motion_model.required_state_types
        required_state_types = motion_model_required_state_types + ["pred_box"]
        return required_state_types
