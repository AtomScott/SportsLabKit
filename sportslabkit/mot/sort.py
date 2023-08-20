from sportslabkit.logger import logger
from sportslabkit.matching import SimpleMatchingFunction
from sportslabkit.metrics import IoUCMM
from sportslabkit.mot.base import MultiObjectTracker


class SORTTracker(MultiObjectTracker):
    """SORT tracker from https://arxiv.org/pdf/1602.00763.pdf"""

    hparam_search_space = {
        "metric_gate": {"type": "float", "low": 1e-2, "high": 1},
        "t_lost": {"type": "int", "low": 1, "high": 1e3},
    }

    def __init__(
        self,
        detection_model,
        motion_model,
        metric: IoUCMM = IoUCMM(use_pred_box=True),
        metric_gate: float = 1.0,
        t_lost: int = 1,
        t_confirm: int = 5,
    ):
        """
        Initializes the SORT Tracker.

        Args:
            detection_model (Any): The model used for object detection.
            motion_model (Any): The model used for motion prediction.
            metric (IoUCMM, optional): The metric used for matching. Defaults to IoUCMM().
            metric_gate (float, optional): The gating threshold for the metric. Defaults to 1.0.
            t_lost (int, optional): The number of frames a tracklet is allowed to be lost for. Defaults to 1.
            t_confirm (int, optional): The number of frames a tracklet needs to be confirmed. Defaults to 5.
        """
        super().__init__(
            pre_init_args={
                "detection_model": detection_model,
                "motion_model": motion_model,
                "metric": metric,
                "metric_gate": metric_gate,
                "t_lost": t_lost,
                "t_confirm": t_confirm,
            }
        )

    def pre_initialize(
        self,
        detection_model,
        motion_model,
        metric,
        metric_gate,
        t_lost,
        t_confirm,
    ):
        self.detection_model = detection_model
        self.motion_model = motion_model

        self.matching_fn = SimpleMatchingFunction(
            metric=metric,
            gate=metric_gate,
        )
        self.t_lost = t_lost
        self.t_confim = t_confirm

    def update(self, current_frame, tracklets):
        # detect objects using the detection model
        detections = self.detection_model(current_frame)

        # update the motion model with the new detections
        # self.update_tracklets_with_motion_model_predictions
        current_boxes = []
        for i, tracklet in enumerate(tracklets):
            # `predicted_box` should be in form [bbleft, bbtop, bbwidth, bbheight]
            predicted_box = self.motion_model(tracklet)
            tracklet.update_state("pred_box", predicted_box)

            # current_box = tracklet.get_observation("box")
            # current_boxes.append(current_box)
            # tracklet.update_current_observation("box", predicted_box)

        # extract features from the detections
        detections = detections[0].to_list()

        # Use predicted tracklets to match with detections since the order is the same
        matches, cost_matrix = self.matching_fn(tracklets, detections, return_cost_matrix=True)

        #
        # for i, tracklet in enumerate(tracklets):
        #     tracklet.update_current_observation("box", current_boxes[i])

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
            tracklet.update_state("staleness", 0)  # reset staleness
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

        # unassigned tracklets: delete if staleness > t_lost
        for i, tracklet in enumerate(tracklets):
            if i not in [match[0] for match in matches]:
                if tracklet.staleness > self.t_lost:
                    tracklet.cleanup()  # remove most recent n=staleness observations
                    unassigned_tracklets.append(tracklet)
                else:
                    new_observation = {
                        "box": tracklet.get_state("pred_box"),
                        "score": tracklet.get_observation("score"),
                        "frame": self.frame_count,
                    }
                    tracklet = self.update_tracklet(tracklet, new_observation)
                    tracklet.staleness += 1
                    assigned_tracklets.append(tracklet)

        return assigned_tracklets, new_tracklets, unassigned_tracklets

    def post_track(self):
        for i, _ in enumerate(self.tracklets):
            self.tracklets[i].cleanup()

        # remove tracklets that a shorter than t_confirm
        for tracklets in [self.tracklets, self.dead_tracklets]:
            confirmed_tracklets = []
            unconfirmed_tracklets = []
            for tracklet in tracklets:
                if len(tracklet) < self.t_confim:
                    unconfirmed_tracklets.append(tracklet)
                else:
                    confirmed_tracklets.append(tracklet)
            tracklets = confirmed_tracklets
        logger.debug(f"Removed {len(unconfirmed_tracklets)} tracklets shorter than {self.t_confim} frames.")

    @property
    def required_observation_types(self):
        return ["box", "frame", "score"]

    @property
    def required_state_types(self):
        motion_model_required_state_types = self.motion_model.required_state_types
        required_state_types = motion_model_required_state_types + ["staleness", "pred_box"]
        return required_state_types
