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
        metric=IoUCMM(),
        metric_gate=1.0,
        t_lost=1,
        t_confirm=5,
    ):
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

    def post_track(self):
        # remove tracklets that a shorter than t_confirm
        for tracklets in [self.tracklets, self.dead_tracklets]:
            confirmed_tracklets = []
            unconfirmed_tracklets = []
            for tracklet in tracklets:
                if tracklet.steps_alive < self.t_confim:
                    unconfirmed_tracklets.append(tracklet)
                else:
                    confirmed_tracklets.append(tracklet)
            tracklets = confirmed_tracklets

    @property
    def required_observation_types(self):
        return ["box", "frame", "score"]

    @property
    def required_state_types(self):
        motion_model_required_state_types = self.motion_model.required_state_types
        required_state_types = motion_model_required_state_types + ["staleness"]
        return required_state_types
