import cv2
import torch
import numpy as np
import soccertrack as st
from soccertrack.types import Tracklet
from soccertrack.mot.base import MultiObjectTracker
from soccertrack.matching import SimpleMatchingFunction, MotionVisualMatchingFunction
from soccertrack.motion_model import KalmanFilterMotionModel
from soccertrack.metrics import IoUCMM, CosineCMM, EuclideanCMM2D
from soccertrack.logger import logger


class TeamTracker(MultiObjectTracker):
    """TeamTrack"""

    def __init__(
        self,
        detection_model=None,
        image_model=None,
        motion_model=None,
        first_matching_fn=MotionVisualMatchingFunction(
            motion_metric=EuclideanCMM2D(),
            motion_metric_beta=0.5,
            motion_metric_gate=0.9,
            visual_metric=CosineCMM(),
            visual_metric_beta=0.5,
            visual_metric_gate=0.5,
        ),
        second_matching_fn=SimpleMatchingFunction(
            metric=EuclideanCMM2D(),
            gate=0.9,
        ),
        conf=0.6,
        t_lost=1,
        H=None,
        multi_target_motion_model=False,
    ):
        super().__init__(
            pre_init_args={
                "detection_model": detection_model,
                "motion_model": motion_model,
                "image_model": image_model,
                "first_matching_fn": first_matching_fn,
                "second_matching_fn": second_matching_fn,
                "conf": conf,
                "t_lost": t_lost,
                "H": H,
                "multi_target_motion_model": multi_target_motion_model,
            }
        )

    def pre_initialize(
        self,
        detection_model,
        motion_model,
        image_model,
        first_matching_fn,
        second_matching_fn,
        conf,
        t_lost,
        H,
        multi_target_motion_model,
    ):
        if detection_model is None:
            # use yolov8 as default
            detection_model = st.detection_model.load("yolov8x")
        self.detection_model = detection_model

        if motion_model is None:
            motion_model = KalmanFilterMotionModel(dt=1 / 30, process_noise=0.1, measurement_noise=0.1)
        self.motion_model = motion_model
        self.motion_model.eval()

        if image_model is None:
            # use osnet as default
            image_model = st.image_model.load("osnet_x1_0")
        self.image_model = image_model

        self.first_matching_fn = first_matching_fn
        self.second_matching_fn = second_matching_fn
        self.conf = conf
        self.t_lost = t_lost
        self.H = H
        self.multi_target_motion_model = multi_target_motion_model

    def update(self, current_frame, tracklets):
        # detect objects using the detection model
        detections = self.detection_model(current_frame)
        # extract features from the detections
        detections = detections[0].to_list()

        # calculate 2d pitch coordinates
        H = self.H
        for i, det in enumerate(detections):
            # calculate the bottom center of the box
            box = det.box
            bcx, bcy = box[0] + box[2] / 2, box[1] + box[3]
            pitch_coordinates = cv2.perspectiveTransform(np.array([[[bcx, bcy]]], dtype="float32"), H)[0][0]
            det.pitch_coordinates = pitch_coordinates

        # update the motion model with the new detections
        # self.update_tracklets_with_motion_model_predictions
        if self.multi_target_motion_model:
            X = []
        for i, tracklet in enumerate(tracklets):
            # calculate the 2d pitch coordinates from the tracklet data
            boxes = np.array(tracklet.get_observations("box"))
            bcxs, bcys = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3]
            x = cv2.perspectiveTransform(np.stack([bcxs, bcys], axis=1).reshape(1, -1, 2).astype("float32"), self.H)[0]
            x = np.expand_dims(x, axis=0)
            x = torch.from_numpy(x)

            # calculate the motion model prediction
            if not self.multi_target_motion_model:
                with torch.no_grad():
                    y = self.motion_model(x).squeeze(0).numpy()
                tracklet.update_state("pitch_coordinates", y)
            else:
                obs_len = self.motion_model.model.input_channels // 2
                if x.shape[1] < obs_len:
                    x = torch.cat([x] + [x[:, 0, :].unsqueeze(1)] * (obs_len - x.shape[1]), dim=1)
                else:
                    x = x[:, -obs_len:]
                X.append(x)
        if self.multi_target_motion_model and len(X) > 0:
            X = torch.stack(X, dim=2)
            with torch.no_grad():
                Y = self.motion_model(X).numpy().squeeze(0)
            for i, tracklet in enumerate(tracklets):
                tracklet.update_state("pitch_coordinates", Y[i])

        if len(detections) > 0:
            embeds = self.image_model.embed_detections(detections, current_frame)
            for i, det in enumerate(detections):
                det.feature = embeds[i]

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

        # [First] Associatie between all tracklets and high confidence detections
        matches_first, cost_matrix_first = self.first_matching_fn(tracklets, high_confidence_detections, True)

        # [First] assigned tracklets: update
        for match in matches_first:
            track_idx, det_idx = match[0], match[1]
            logger.debug(f"track_idx: {track_idx}, det_idx: {det_idx}, cost: {cost_matrix_first[track_idx, det_idx]}")
            det = high_confidence_detections[det_idx]
            tracklet = tracklets[track_idx]

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
