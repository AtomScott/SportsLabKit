import time
import uuid
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from motpy.core import Box, Detection, Track, Vector
from motpy.model import Model, ModelPreset

from sportslabkit import BBoxDataFrame
from sportslabkit.logger import logger


def get_kalman_object_tracker(
    model: Model, x0: Optional[Vector] = None
) -> KalmanFilter:
    """returns Kalman-based tracker based on a specified motion model spec.
    e.g. for spec = {'order_pos': 1, 'dim_pos': 2, 'order_size': 0, 'dim_size': 1}
    we expect the following setup:
    state x, x', y, y', w, h
    where x and y are centers of boxes
          w and h are width and height
    """

    tracker = KalmanFilter(dim_x=model.state_length, dim_z=model.measurement_length)
    tracker.F = model.build_F()
    tracker.Q = model.build_Q()
    tracker.H = model.build_H()
    tracker.R = model.build_R()
    tracker.P = model.build_P()

    if x0 is not None:
        tracker.x = x0

    return tracker


DEFAULT_MODEL_SPEC = ModelPreset.constant_velocity_and_static_box_size_2d.value


def exponential_moving_average_fn(gamma: float) -> Callable:
    def fn(old, new):
        if new is None:
            return old

        if isinstance(new, Iterable):
            new = np.array(new)

        if old is None:
            return new  # first call

        if isinstance(old, Iterable):
            old = np.array(old)

        return gamma * old + (1 - gamma) * new

    return fn


class SingleObjectTracker:
    def __init__(
        self,
        max_staleness: float = 12.0,
        smooth_score_gamma: float = 0.8,
        smooth_feature_gamma: float = 0.9,
        score0: Optional[float] = None,
        class_id0: Optional[int] = None,
    ):
        """
        Args:
            max_staleness (float, optional): number of steps after which a tracker is considered stale. Defaults to 12.0.
            smooth_score_gamma (float, optional): smoothing factor for score. Defaults to 0.8.
            smooth_feature_gamma (float, optional): smoothing factor for feature. Defaults to 0.9.
            score0 (Optional[float], optional): initial score. Defaults to None.
            class_id0 (Optional[int], optional): initial class id. Defaults to None.
        """

        self.id: int = int(str(int(uuid.uuid4()))[:12])
        self.steps_alive: int = 1
        self.steps_positive: int = 1
        self.staleness: float = 0.0
        self.max_staleness: float = max_staleness

        self.update_score_fn: Callable = exponential_moving_average_fn(
            smooth_score_gamma
        )
        self.update_feature_fn: Callable = exponential_moving_average_fn(
            smooth_feature_gamma
        )

        self.score: Optional[float] = score0
        self.feature: Optional[Vector] = None

        self.class_id_counts: Dict = dict()
        self.class_id: Optional[int] = self.update_class_id(class_id0)
        self.states = []

        logger.debug(f"creating new tracker {self.id}")

    def box(self) -> Box:
        raise NotImplementedError()

    def is_invalid(self) -> bool:
        raise NotImplementedError()

    def _predict(self) -> None:
        raise NotImplementedError()

    def predict(self) -> None:
        self._predict()
        self.steps_alive += 1

    def update_class_id(self, class_id: Optional[int]) -> Optional[int]:
        """find most frequent prediction of class_id in recent K class_ids"""
        if class_id is None:
            return None

        if class_id in self.class_id_counts:
            self.class_id_counts[class_id] += 1
        else:
            self.class_id_counts[class_id] = 1

        return max(self.class_id_counts, key=self.class_id_counts.get)

    def _update_box(self, detection: Detection) -> None:
        raise NotImplementedError()

    def update(self, detection: Detection, global_step: Optional[int] = None) -> None:
        """update the tracker with a new detection

        Args:
            detection (Detection): detection to update the tracker with
            global_step (Optional[int], optional): global step of the detection. Defaults to None.
        """
        self._update_box(detection)

        self.steps_positive += 1

        self.class_id = self.update_class_id(detection.class_id)
        self.score = self.update_score_fn(old=self.score, new=detection.score)
        self.feature = self.update_feature_fn(old=self.feature, new=detection.feature)

        # reduce the staleness of a tracker, faster than growth rate
        self.unstale(rate=3)

        # save the state of the tracker
        self.global_step = global_step
        self.save_state()

    def save_state(self) -> None:
        """save the state of the tracker"""
        self.state = {
            "id": self.id,
            "steps_alive": self.steps_alive,
            "steps_positive": self.steps_positive,
            "staleness": self.staleness,
            "score": self.score,
            "feature": self.feature,
            "class_id": self.class_id,
            "class_id_counts": self.class_id_counts,
            "box": self.box(),
            "global_step": self.global_step,
        }
        self.states.append(self.state)

    def stale(self, rate: float = 1.0) -> float:
        self.staleness += rate
        return self.staleness

    def unstale(self, rate: float = 2.0) -> float:
        self.staleness = max(0, self.staleness - rate)
        return self.staleness

    def is_stale(self) -> bool:
        return self.staleness >= self.max_staleness

    def to_bbdf(self):
        states = self.states
        if len(states) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(states)[["global_step", "box", "id", "score"]]

        df = pd.DataFrame(
            df["box"].to_list(), columns=["bb_left", "bb_top", "bb_width", "bb_height"]
        ).join(df.drop(columns=["box"]))

        df.rename(columns={"global_step": "frame", "score": "conf"}, inplace=True)

        df.set_index(["frame"], inplace=True)

        box_df = df[["bb_left", "bb_top", "bb_width", "bb_height", "conf"]]
        team_id = 0
        player_id = df.id.unique()[0]

        idx = pd.MultiIndex.from_product(
            [[team_id], [player_id], box_df.columns],
            names=["TeamID", "PlayerID", "Attributes"],
        )

        return BBoxDataFrame(box_df.values, index=df.index, columns=idx)

    def __repr__(self) -> str:
        return f"(box: {str(self.box())}, score: {self.score}, class_id: {self.class_id}, staleness: {self.staleness:.2f})"


class KalmanTracker(SingleObjectTracker):
    """A single object tracker using Kalman filter with specified motion model specification"""

    def __init__(
        self,
        model_kwargs: dict = DEFAULT_MODEL_SPEC,
        x0: Optional[Vector] = None,
        box0: Optional[Box] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.model_kwargs: dict = model_kwargs
        self.model = Model(**self.model_kwargs)
        if x0 is None:
            x0 = self.model.box_to_x(box0)

        self._tracker: KalmanFilter = get_kalman_object_tracker(model=self.model, x0=x0)

    def _predict(self) -> None:
        self._tracker.predict()

    def _update_box(self, detection: Detection) -> None:
        z = self.model.box_to_z(detection.box)
        self._tracker.update(z)

    def box(self) -> Box:
        return self.model.x_to_box(self._tracker.x)

    def is_invalid(self) -> bool:
        try:
            has_nans = any(np.isnan(self._tracker.x))
            return has_nans
        except Exception as e:
            logger.warning(f"invalid tracker - exception: {e}")
            return True


# class MultiObjectTracker:
#     def __init__(
#         self,
#         dt: float,
#         model_spec: Union[str, Dict] = DEFAULT_MODEL_SPEC,
#         matching_fn: Optional[Any] = None,
#         tracker_kwargs: Dict = None,
#         matching_fn_kwargs: Dict = None,
#         active_tracks_kwargs: Dict = None,
#     ) -> None:
#         """
#         model_spec specifies the dimension and order for position and size of the object
#         matching_fn determines the strategy on which the trackers and detections are assigned.

#         tracker_kwargs are passed to each single object tracker
#         active_tracks_kwargs limits surfacing of fresh/fading out tracks
#         """

#         self.trackers: List[SingleObjectTracker] = []
#         self.stale_trackers: List[SingleObjectTracker] = []

#         # kwargs to be passed to each single object tracker
#         self.tracker_kwargs: Dict = tracker_kwargs if tracker_kwargs is not None else {}
#         self.tracker_clss: Optional[Type[SingleObjectTracker]] = None

#         # translate model specification into single object tracker to be used
#         if isinstance(model_spec, dict):
#             self.tracker_clss = KalmanTracker
#             self.tracker_kwargs["model_kwargs"] = model_spec
#             self.tracker_kwargs["model_kwargs"]["dt"] = dt
#         elif isinstance(model_spec, str) and model_spec in ModelPreset.__members__:
#             self.tracker_clss = KalmanTracker
#             self.tracker_kwargs["model_kwargs"] = ModelPreset[model_spec].value
#             self.tracker_kwargs["model_kwargs"]["dt"] = dt
#         else:
#             raise NotImplementedError(f"unsupported motion model {model_spec}")

#         logger.debug(
#             f"using single tracker of class: {self.tracker_clss} with kwargs: {self.tracker_kwargs}"
#         )

#         self.matching_fn: Any = matching_fn
#         self.matching_fn_kwargs: Dict = (
#             matching_fn_kwargs if matching_fn_kwargs is not None else {}
#         )
#         # if self.matching_fn is None:
#         #     self.matching_fn = IOUAndFeatureMatchingFunction(**self.matching_fn_kwargs)

#         # kwargs to be used when self.step returns active tracks
#         self.active_tracks_kwargs: Dict = (
#             active_tracks_kwargs if active_tracks_kwargs is not None else {}
#         )
#         logger.debug("using active_tracks_kwargs: %s" % str(self.active_tracks_kwargs))

#         self.detections_matched_ids = []
#         self.current_step = 0

#     def active_tracks(
#         self,
#         max_staleness_to_positive_ratio: float = 3.0,
#         max_staleness: float = 999,
#         min_steps_alive: int = -1,
#     ) -> List[Track]:
#         """returns all active tracks after optional filtering by tracker steps count and staleness"""

#         tracks: List[Track] = []
#         for tracker in self.trackers:
#             cond1 = (
#                 tracker.staleness / tracker.steps_positive
#                 < max_staleness_to_positive_ratio
#             )  # early stage
#             cond2 = tracker.staleness < max_staleness
#             cond3 = tracker.steps_alive >= min_steps_alive
#             if cond1 and cond2 and cond3:
#                 tracks.append(
#                     Track(
#                         id=tracker.id,
#                         box=tracker.box(),
#                         score=tracker.score,
#                         class_id=tracker.class_id,
#                     )
#                 )

#         logger.debug("active/all tracks: %d/%d" % (len(self.trackers), len(tracks)))
#         return tracks

#     def all_trackers(self) -> List[Track]:
#         """returns all trackers"""
#         return self.trackers + self.stale_trackers

#     def to_bbdf(self):
#         """Create a bounding box dataframe"""

#         return pd.concat([t.to_bbdf() for t in self.all_trackers()], axis=1)

#     def cleanup_trackers(self) -> None:
#         """Moves stale trackers into the stale_trackers list"""
#         count_before = len(self.trackers)
#         self.stale_trackers.extend(
#             [t for t in self.trackers if t.is_stale() and not t.is_invalid()]
#         )
#         self.trackers = [
#             t for t in self.trackers if not (t.is_stale() or t.is_invalid())
#         ]
#         count_after = len(self.trackers)
#         logger.debug(
#             "deleted %s/%s trackers" % (count_before - count_after, count_before)
#         )

#     def step(self, detections: Sequence[Detection]) -> List[Track]:
#         """the method matches the new detections with existing trackers,
#         creates new trackers if necessary and performs the cleanup.
#         Returns the active tracks after active filtering applied"""
#         t0 = time.time()

#         # filter out empty detections
#         detections = [det for det in detections if det.box is not None]

#         # predict state in all trackers
#         for t in self.trackers:
#             t.predict()

#         # match trackers with detections
#         logger.debug("step with %d detections" % len(detections))
#         matches = self.matching_fn(self.trackers, detections)
#         logger.debug("matched %d pairs" % len(matches))

#         self.detections_matched_ids = [None] * len(detections)

#         # assigned trackers: correct
#         for match in matches:
#             track_idx, det_idx = match[0], match[1]
#             self.trackers[track_idx].update(
#                 detection=detections[det_idx], global_step=self.current_step
#             )
#             self.detections_matched_ids[det_idx] = self.trackers[track_idx].id

#         # not assigned detections: create new trackers POF
#         assigned_det_idxs = set(matches[:, 1]) if len(matches) > 0 else []
#         for det_idx in set(range(len(detections))).difference(assigned_det_idxs):
#             det = detections[det_idx]
#             tracker = self.tracker_clss(
#                 box0=det.box,
#                 score0=det.score,
#                 class_id0=det.class_id,
#                 **self.tracker_kwargs,
#             )
#             self.detections_matched_ids[det_idx] = tracker.id
#             self.trackers.append(tracker)

#         # unassigned trackers
#         assigned_track_idxs = set(matches[:, 0]) if len(matches) > 0 else []
#         for track_idx in set(range(len(self.trackers))).difference(assigned_track_idxs):
#             self.trackers[track_idx].stale()

#         # cleanup dead trackers
#         self.cleanup_trackers()

#         # log step timing
#         elapsed = (time.time() - t0) * 1000.0
#         logger.debug(
#             f"tracking step time: {elapsed:.3f} ms @ current step: {self.current_step}"
#         )

#         # update current step
#         self.current_step += 1

#         return self.active_tracks(**self.active_tracks_kwargs)
