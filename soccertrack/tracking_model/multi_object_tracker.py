from __future__ import annotations

import time
import uuid
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from motpy.model import Model, ModelPreset

from soccertrack import BBoxDataFrame, Camera
from soccertrack.logger import logger, tqdm
from soccertrack.tracking_model.tracker import KalmanTracker, SingleObjectTracker
from soccertrack.types import Box, Detection, Track, Vector, _pathlike

DEFAULT_MODEL_SPEC = ModelPreset.constant_velocity_and_static_box_size_2d.value


class MultiObjectTracker:
    """The main component that manages the tracking of multiple objects.

    Args:
        dt: Time step in seconds.
        model_spec: Specifies the dimension and order for position and size of the object.
        matching_fn: Determines the strategy on which the trackers and detections are assigned.
        tracker_kwarg: Are passed to each single object tracker.
        active_tracks_kwargs: Limits surfacing of fresh/fading out tracks.

    Examples:
        >>> mot = MultiObjectTracker(dt=1/25, model_spec="constant_velocity_and_static_box_size_2d")
        >>> res = mot.track("path/to/video.mp4")
        >>> res.to_bbdf()
    """

    def __init__(
        self,
        detection_model=None,
        image_model=None,
        motion_model=None,
        matching_fn: Any | None = None,
        # matching_fn_kwargs: dict = None,
        # dt: float = None,
        # model_spec: str | dict = DEFAULT_MODEL_SPEC,
        # matching_fn: Any | None = None,
        # tracker_kwargs: dict = None,
        # active_tracks_kwargs: dict = None,
    ) -> None:

        self.detection_model = detection_model
        self.image_model = image_model
        self.motion_model = motion_model
        self.matching_fn: Any = matching_fn

        self.trackers: list[SingleObjectTracker] = []
        self.stale_trackers: list[SingleObjectTracker] = []

        # kwargs to be passed to each single object tracker
        # self.tracker_kwargs: dict = tracker_kwargs if tracker_kwargs is not None else {}
        # self.tracker_clss: SingleObjectTracker = None

        # translate model specification into single object tracker to be used
        # if isinstance(model_spec, dict):
        #     self.tracker_clss = KalmanTracker
        #     self.tracker_kwargs["model_kwargs"] = model_spec
        #     self.tracker_kwargs["model_kwargs"]["dt"] = dt
        # elif isinstance(model_spec, str) and model_spec in ModelPreset.__members__:
        #     self.tracker_clss = KalmanTracker
        #     self.tracker_kwargs["model_kwargs"] = ModelPreset[model_spec].value
        #     self.tracker_kwargs["model_kwargs"]["dt"] = dt
        # else:
        #     raise NotImplementedError(f"unsupported motion model {model_spec}")

        # logger.debug(
        #     f"using single tracker of class: {self.tracker_clss} with kwargs: {self.tracker_kwargs}"
        # )

        # self.matching_fn_kwargs: dict = (
        #     matching_fn_kwargs if matching_fn_kwargs is not None else {}
        # )
        # if self.matching_fn is None:
        #     self.matching_fn = IOUAndFeatureMatchingFunction(**self.matching_fn_kwargs)

        # kwargs to be used when self.step returns active tracks
        # self.active_tracks_kwargs: dict = (
        #     active_tracks_kwargs if active_tracks_kwargs is not None else {}
        # )
        # logger.debug("using active_tracks_kwargs: %s" % str(self.active_tracks_kwargs))

        self.detections_matched_ids = []
        self.current_step = 0

    def track(self, source: _pathlike | Camera):
        """Tracks objects in a video.

        Args:
            source: Path to a video file or a Camera object.

        Returns:
            The result of the tracking as a xxx.
        """

        if isinstance(source, str):
            cam = Camera(source)
        if not isinstance(source, Camera):
            cam = Camera(source)

        dets = []
        for frame in (pbar := tqdm(cam.iter_frames())):

            # detect objects using the detection model
            detections = self.det_model(frame).to_list()
            dets.append(detections)

            # update the state of the multi-object-tracker tracker
            # with the list of bounding boxes
            self.step(detections=detections)

            # get tracks to be displayed
            tracks = self.all_tracks()
            pbar.set_postfix({f"Number of active tracks": len(tracks)})

        result = None
        return result

    def step(self, detections: list[Detection]) -> list[Track]:
        """the method matches the new detections with existing trackers,
        creates new trackers if necessary and performs the cleanup.

        Returns the active tracks after active filtering applied
        """
        # filter out empty detections
        # detections = [det for det in detections if det.box is not None]

        # predict state in all trackers
        for t in self.trackers:
            t.predict()

        # match trackers with detections
        logger.debug("step with %d detections" % len(detections))
        matches = self.matching_fn(self.trackers, detections)
        logger.debug("matched %d pairs" % len(matches))

        self.detections_matched_ids = [None] * len(detections)

        # assigned trackers: correct
        for match in matches:
            track_idx, det_idx = match[0], match[1]
            self.trackers[track_idx].update(
                detection=detections[det_idx], global_step=self.current_step
            )
            self.detections_matched_ids[det_idx] = self.trackers[track_idx].id

        # not assigned detections: create new trackers POF
        assigned_det_idxs = set(matches[:, 1]) if len(matches) > 0 else []
        for det_idx in set(range(len(detections))).difference(assigned_det_idxs):
            det = detections[det_idx]
            tracker = self.tracker_clss(
                box0=det.box,
                score0=det.score,
                class_id0=det.class_id,
                **self.tracker_kwargs,
            )
            self.detections_matched_ids[det_idx] = tracker.id
            self.trackers.append(tracker)

        # unassigned trackers
        assigned_track_idxs = set(matches[:, 0]) if len(matches) > 0 else []
        for track_idx in set(range(len(self.trackers))).difference(assigned_track_idxs):
            self.trackers[track_idx].stale()

        # cleanup dead trackers
        self.cleanup_trackers()

        # update current step
        self.current_step += 1

        return self.active_tracks(**self.active_tracks_kwargs)

    def active_tracks(
        self,
        max_staleness_to_positive_ratio: float = 3.0,
        max_staleness: float = 999,
        min_steps_alive: int = -1,
    ) -> list[Track]:
        """returns all active tracks after optional filtering by tracker steps
        count and staleness."""

        tracks: list[Track] = []
        for tracker in self.trackers:
            cond1 = (
                tracker.staleness / tracker.steps_positive
                < max_staleness_to_positive_ratio
            )  # early stage
            cond2 = tracker.staleness < max_staleness
            cond3 = tracker.steps_alive >= min_steps_alive
            if cond1 and cond2 and cond3:
                tracks.append(
                    Track(
                        id=tracker.id,
                        box=tracker.box(),
                        score=tracker.score,
                        class_id=tracker.class_id,
                    )
                )

        logger.debug("active/all tracks: %d/%d" % (len(self.trackers), len(tracks)))
        return tracks

    def all_trackers(self) -> list[Track]:
        """returns all trackers."""
        return self.trackers + self.stale_trackers

    def to_bbdf(self):
        """Create a bounding box dataframe."""

        return pd.concat([t.to_bbdf() for t in self.all_trackers()], axis=1)

    def cleanup_trackers(self) -> None:
        """Moves stale trackers into the stale_trackers list."""
        count_before = len(self.trackers)
        self.stale_trackers.extend(
            [t for t in self.trackers if t.is_stale() and not t.is_invalid()]
        )
        self.trackers = [
            t for t in self.trackers if not (t.is_stale() or t.is_invalid())
        ]
        count_after = len(self.trackers)
        logger.debug(
            "deleted %s/%s trackers" % (count_before - count_after, count_before)
        )
