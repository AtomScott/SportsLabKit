import cv2
import numpy as np
import torch

from sportslabkit.logger import logger
from sportslabkit.matching import MotionVisualMatchingFunction, SimpleMatchingFunction
from sportslabkit.metrics import CosineCMM, IoUCMM
from sportslabkit.mot.base import MultiObjectTracker


class TeamTracker(MultiObjectTracker):
    """TeamTrack"""

    def __init__(
        self,
        detection_model=None,
        image_model=None,
        motion_model=None,
        calibration_model=None,
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
        step_size: int | None = None,
        max_staleness: int = 5,
        min_length: int = 5,
        callbacks=None,
    ):
        super().__init__(
            window_size=window_size,
            step_size=step_size,
            max_staleness=max_staleness,
            min_length=min_length,
            callbacks=callbacks,
        )
        self.detection_model = detection_model
        self.image_model = image_model
        self.motion_model = motion_model
        self.calibration_model = calibration_model
        self.first_matching_fn = first_matching_fn
        self.second_matching_fn = second_matching_fn
        self.detection_score_threshold = detection_score_threshold
        self.homographies = []

    def predict_single_tracklet_motion(self, tracklet):
        # x = self.tracklet_to_points(tracklet, H)
        y = self.motion_model(tracklet).squeeze().numpy()
        return y

    def predict_multi_tracklet_motion(self, tracklets):
        # for i, tracklet in enumerate(tracklets):
        # x = self.tracklet_to_points(tracklet)
        # X.append(tracklet)

        with torch.no_grad():
            Y = self.motion_model(tracklets)
            Y = np.array(Y).squeeze()
        return Y
        # obs_len = self.motion_model.model.input_channels // 2
        # if x.shape[1] < obs_len:
        #     x = torch.cat([x] + [x[:, 0, :].unsqueeze(1)] * (obs_len - x.shape[1]), dim=1)
        # else:
        #     x = x[:, -obs_len:]
        # X.append(x)

    # if self.multi_target_motion_model and len(X) > 0:
    #     X = torch.stack(X, dim=2)
    #     with torch.no_grad():
    #         Y = self.motion_model(X).numpy().squeeze(0)
    #     for i, tracklet in enumerate(tracklets):
    #         tracklet.update_state("pitch_coordinates", Y[i])

    def tracklet_to_points(self, tracklet, H):
        # calculate the 2d pitch coordinates from the tracklet data
        boxes = np.array(tracklet.get_observations("box"))
        bcxs, bcys = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3]
        x = cv2.perspectiveTransform(np.stack([bcxs, bcys], axis=1).reshape(1, -1, 2).astype("float32"), H)[0]
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x)
        return x

    def detection_to_points(self, detection, H):
        box = detection.box
        bcx, bcy = box[0] + box[2] / 2, box[1] + box[3]
        return cv2.perspectiveTransform(np.array([[[bcx, bcy]]], dtype="float32"), H)[0][0]

    def update(self, current_frame, tracklets):
        # detect objects using the detection model
        detections = self.detection_model(current_frame)

        # extract features from the detections
        detections = detections[0].to_list()

        # calculate 2d pitch coordinates
        H = self.calibration_model(current_frame)
        self.homographies.append(H)

        dets_ids_to_remove = []
        for i, det in enumerate(detections):
            det.pt = self.detection_to_points(det, H)

            # remove detections that are outside the pitch
            # add other sports
            if det.pt[0] < 0 or det.pt[0] > 105 or det.pt[1] < 0 or det.pt[1] > 68:
                dets_ids_to_remove.append(i)

        for i in sorted(dets_ids_to_remove, reverse=True):
            del detections[i]

        ##############################
        # Motion prediction
        ##############################
        # `pred_pt` should be in form [x, y]
        # `pred_pts` should be in form [n, x, y]
        if self.motion_model.is_multi_target:
            pred_pts = self.predict_multi_tracklet_motion(tracklets)

        for i, tracklet in enumerate(tracklets):
            if self.motion_model.is_multi_target:
                pred_pt = pred_pts[i]
            else:
                pred_pt = self.predict_single_tracklet_motion(tracklet)
            tracklet.update_state("pred_pt", pred_pt)

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

            new_state = {
                "box": det.box,
                "pt": det.pt,
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
                    "pt": det.pt,
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
        matches_second, cost_matrix_second = self.second_matching_fn(unassigned_tracklets, low_confidence_detections, True)

        # [Second] assigned tracklets: update
        for match in matches_second:
            track_idx, det_idx = match[0], match[1]
            logger.debug(f"track_idx: {track_idx}, det_idx: {det_idx}, cost: {cost_matrix_second[track_idx, det_idx]}")

            det = low_confidence_detections[det_idx]
            tracklet = unassigned_tracklets[track_idx]

            new_state = {
                "box": det.box,
                "pt": det.pt,
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
                new_observation = {
                    "box": tracklet.get_state("pred_box"),
                    "score": tracklet.get_observation("score"),
                    "frame": self.frame_count,
                    "feature": tracklet.get_observation("feature"),
                    "pt": tracklet.get_state("pred_pt"),
                }
                tracklet = self.update_tracklet(tracklet, new_observation)
                unassigned_tracklets_second.append(tracklet)

        logger.debug(f"1st matches: {len(matches_first)}, 2nd matches: {len(matches_second)}")
        return assigned_tracklets, new_tracklets, unassigned_tracklets_second

    @property
    def required_observation_types(self):
        return ["box", "frame", "score", "feature", "pt"]

    @property
    def required_state_types(self):
        motion_model_required_state_types = self.motion_model.required_state_types
        required_state_types = motion_model_required_state_types + ["pred_pt"]
        return required_state_types
