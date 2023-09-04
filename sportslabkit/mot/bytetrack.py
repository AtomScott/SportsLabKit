from typing import Optional

from sportslabkit.logger import logger
from sportslabkit.matching import MotionVisualMatchingFunction, SimpleMatchingFunction
from sportslabkit.metrics import CosineCMM, IoUCMM
from sportslabkit.mot.base import MultiObjectTracker


class BYTETracker(MultiObjectTracker):
    """BYTE tracker from https://arxiv.org/pdf/2110.06864.pdf"""

    def __init__(
        self,
        detection_model=None,
        image_model=None,
        motion_model=None,
        first_matching_fn: MotionVisualMatchingFunction = MotionVisualMatchingFunction(
            motion_metric=IoUCMM(use_pred_box=True),
            motion_metric_gate=0.2,
            visual_metric=CosineCMM(),
            visual_metric_gate=0.2,
            beta=0.5,
        ),
        second_matching_fn=SimpleMatchingFunction(
            metric=IoUCMM(use_pred_box=True),
            gate=0.9,
        ),
        detection_score_threshold=0.6,
        window_size: int = 1,
        step_size: Optional[int] = None,
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
        self.first_matching_fn = first_matching_fn
        self.second_matching_fn = second_matching_fn
        self.detection_score_threshold = detection_score_threshold

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

        # separate the detections into high and low confidence
        high_confidence_detections = []
        low_confidence_detections = []
        for detection in detections:
            if detection.score > self.detection_score_threshold:
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

        # [First] Associatie between all tracklets and high confidence detections
        matches_first, cost_matrix_first = self.first_matching_fn(tracklets, high_confidence_detections, True)

        # [First] assigned tracklets: update
        for match in matches_first:
            track_idx, det_idx = match[0], match[1]
            logger.debug(f"track_idx: {track_idx}, det_idx: {det_idx}, cost: {cost_matrix_first[track_idx, det_idx]}")
            det = high_confidence_detections[det_idx]
            tracklet = tracklets[track_idx]

            new_observation = {
                "box": det.box,
                "score": det.score,
                "feature": det.feature,
                "frame": self.frame_count,
            }

            # update the tracklet with the new state
            tracklet = self.update_tracklet(tracklet, new_observation)
            assigned_tracklets.append(tracklet)

        # [First] not assigned detections: create new trackers
        for i, det in enumerate(high_confidence_detections):
            if i not in [match[1] for match in matches_first]:
                new_observation = {
                    "box": det.box,
                    "score": det.score,
                    "frame": self.frame_count,
                    "feature": det.feature,
                }
                new_tracklet = self.create_tracklet(new_observation)
                new_tracklets.append(new_tracklet)

        # [First] unassigned tracklets: store for second association
        for i, tracklet in enumerate(tracklets):
            if i not in [match[0] for match in matches_first]:
                unassigned_tracklets.append(tracklet)

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

            new_observation = {
                "box": det.box,
                "score": det.score,
                "feature": det.feature,
                "frame": self.frame_count,
            }

            # update the tracklet with the new state
            tracklet = self.update_tracklet(tracklet, new_observation)
            assigned_tracklets.append(tracklet)

        # [Second] not assigned detections: Do nothing
        # We don't want to create new tracklets for low confidence detections

        # [Second] Get the tracklets that were not matched
        unassigned_tracklets_second = []
        for i, tracklet in enumerate(unassigned_tracklets):
            if i not in [match[0] for match in matches_second]:
                new_observation = {
                    "box": tracklet.get_state("pred_box"),
                    "score": tracklet.get_observation("score"),
                    "frame": self.frame_count,
                    "feature": tracklet.get_observation("feature"),
                }
                tracklet = self.update_tracklet(tracklet, new_observation)
                unassigned_tracklets_second.append(tracklet)

        logger.debug(f"1st matches: {len(matches_first)}, 2nd matches: {len(matches_second)}")
        return assigned_tracklets, new_tracklets, unassigned_tracklets_second

    @property
    def required_observation_types(self):
        return ["box", "frame", "score", "feature"]

    @property
    def required_state_types(self):
        motion_model_required_state_types = self.motion_model.required_state_types
        required_state_types = motion_model_required_state_types + ["pred_box"]
        return required_state_types
