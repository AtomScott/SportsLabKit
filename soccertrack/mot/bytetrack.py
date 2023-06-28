import soccertrack as st
from soccertrack.types import Tracklet
from soccertrack.mot.base import MultiObjectTracker
from soccertrack.matching import SimpleMatchingFunction, MotionVisualMatchingFunction
from soccertrack.motion_model import KalmanFilterMotionModel
from soccertrack.metrics import IoUCMM, CosineCMM
from soccertrack.logger import logger


class BYTETracker(MultiObjectTracker):
    """BYTE tracker from https://arxiv.org/pdf/2110.06864.pdf"""

    def __init__(
        self,
        detection_model=None,
        motion_model=None,
        first_matching_fn=MotionVisualMatchingFunction(
            motion_metric=IoUCMM(),
            motion_metric_beta=0.5,
            motion_metric_gate=0.9,
            visual_metric=CosineCMM(),
            visual_metric_beta=0.5,
            visual_metric_gate=0.5,
        ),
        second_matching_fn=SimpleMatchingFunction(
            metric=IoUCMM(),
            gate=0.9,
        ),
        conf=0.6,
        t_lost=1,
    ):
        super().__init__(
            pre_init_args={
                "detection_model": detection_model,
                "motion_model": motion_model,
                "first_matching_fn": first_matching_fn,
                "second_matching_fn": second_matching_fn,
                "conf": conf,
                "t_lost": t_lost,
            }
        )

    def pre_initialize(
        self,
        detection_model,
        motion_model,
        first_matching_fn,
        second_matching_fn,
        conf,
        t_lost,
    ):
        if detection_model is None:
            # use yolov8 as default
            detection_model = st.detection_model.load("yolov8x")
        self.detection_model = detection_model

        if motion_model is None:
            motion_model = KalmanFilterMotionModel(dt=1 / 30, process_noise=0.1, measurement_noise=0.1)
        self.motion_model = motion_model

        self.first_matching_fn = first_matching_fn
        self.second_matching_fn = second_matching_fn
        self.conf = conf
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

        # separate the detections into high and low confidence
        high_confidence_detections = []
        low_confidence_detections = []
        for detection in detections:
            if detection.score > self.conf:
                high_confidence_detections.append(detection)
            else:
                low_confidence_detections.append(detection)
        logger.debug(f"d_high: {len(high_confidence_detections)}, d_low: {len(low_confidence_detections)}")

        ##############################
        # First association
        ##############################
        new_tracklets = []
        assigned_tracklets = []
        unassigned_tracklets = []
        unassigned_current_boxes = []

        # [First] Associatie between all tracklets and high confidence detections
        matches_first, cost_matrix_first = self.first_matching_fn(tracklets, high_confidence_detections, True)

        # for i, tracklet in enumerate(tracklets):
        #     tracklet.update_current_observation("box", current_boxes[i])

        # [First] assigned tracklets: update
        for match in matches_first:
            track_idx, det_idx = match[0], match[1]
            logger.debug(f"track_idx: {track_idx}, det_idx: {det_idx}, cost: {cost_matrix_first[track_idx, det_idx]}")
            det = high_confidence_detections[det_idx]
            tracklet = tracklets[track_idx]

            # Undo previous update that was done with the motion model
            current_box = current_boxes[track_idx]
            tracklet.update_current_observation("box", current_box)

            new_state = {
                "box": det.box,
                "score": det.score,
                "feature": det.feature,
                "frame": self.frame_count,
            }

            # update the tracklet with the new state
            tracklet = self.update_tracklet(tracklet, new_state)
            assigned_tracklets.append(tracklet)

        # [First] not assigned detections: create new trackers
        for i, det in enumerate(high_confidence_detections):
            if i not in [match[1] for match in matches_first]:
                new_state = {
                    "box": det.box,
                    "score": det.score,
                    "frame": self.frame_count,
                    "feature": det.feature,
                }
                new_tracklet = self.create_tracklet(new_state)
                new_tracklets.append(new_tracklet)

        # [First] Get the tracklets that were not matched
        for i, tracklet in enumerate(tracklets):
            if i not in [match[0] for match in matches_first]:
                unassigned_tracklets.append(tracklet)
                unassigned_current_boxes.append(current_boxes[i])

        ##############################
        # Second association
        ##############################

        # Second association between unassigned tracklets and low confidence detections
        matches_second, cost_matrix_second = self.second_matching_fn(
            unassigned_tracklets, low_confidence_detections, True
        )

        # [Second] assigned tracklets: update
        for match in matches_second:
            track_idx, det_idx = match[0], match[1]
            logger.debug(f"track_idx: {track_idx}, det_idx: {det_idx}, cost: {cost_matrix_second[track_idx, det_idx]}")

            det = low_confidence_detections[det_idx]
            tracklet = unassigned_tracklets[track_idx]

            # Undo previous update that was done with the motion model
            current_box = unassigned_current_boxes[track_idx]
            tracklet.update_current_observation("box", current_box)

            new_state = {
                "box": det.box,
                "score": det.score,
                "feature": det.feature,
                "frame": self.frame_count,
            }

            # update the tracklet with the new state
            tracklet = self.update_tracklet(tracklet, new_state)
            assigned_tracklets.append(tracklet)

        # [Second] not assigned detections: Do nothing
        # We don't want to create new tracklets for low confidence detections

        # [Second] Get the tracklets that were not matched
        unassigned_tracklets_second = []
        for i, tracklet in enumerate(unassigned_tracklets):
            if i not in [match[0] for match in matches_second]:
                staleness = tracklet.get_state("staleness")
                if staleness is None:
                    staleness = 0
                if staleness > self.t_lost:
                    unassigned_tracklets_second.append(tracklet)
                else:
                    tracklet.update_state("staleness", staleness + 1)
                    assigned_tracklets.append(tracklet)

        logger.debug(f"1st matches: {len(matches_first)}, 2nd matches: {len(matches_second)}")
        return assigned_tracklets, new_tracklets, unassigned_tracklets_second

    @property
    def required_observation_types(self):
        return ["box", "frame", "score", "feature"]

    @property
    def required_state_types(self):
        motion_model_required_state_types = self.motion_model.required_state_types
        required_state_types = motion_model_required_state_types + ["staleness"]
        return required_state_types
