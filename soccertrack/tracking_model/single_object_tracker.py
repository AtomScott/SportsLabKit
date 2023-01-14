from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd

from soccertrack import BBoxDataFrame
from soccertrack.logger import logger
from soccertrack.types import Box, Detection, Tracker, Vector


class SingleObjectTracker(Tracker):
    """Single object tracker.

    The default single object tracker just keeps track of given detections and returns the most recent one.

    Args:
        max_staleness: number of steps after which a tracker is considered stale. Defaults to 12.0.
        keep_observations: whether to keep track of observations. Defaults to True.
        keep_predictions: whether to keep track of predictions. Defaults to True.

    Attributes:
        id (int): unique id of the tracker
        steps_alive (int): number of steps the tracker was alive
        steps_positive (int): number of steps the tracker was positive (i.e. had a detection associated with it)
        staleness (float): number of steps since the last positive update
        global_step (int): number of steps since the start of the tracking process
        max_staleness (float): number of steps after which a tracker is considered stale
    """

    def __init__(
        self,
        max_staleness: float = 12.0,
        keep_observations: bool = True,
        keep_predictions: bool = True,
    ):

        # initialize states
        self.id: int = int(str(int(uuid.uuid4()))[:12])
        self.steps_alive: int = 0
        self.steps_positive: int = 0
        self.staleness: float = 0.0
        self.global_step: int = 0

        # initialize parameters
        self.max_staleness: float = max_staleness

        # initialize observations
        # these are updated by the update method
        self._obs_boxes: List[Box] = []
        self._obs_scores: List[float] = []
        self._obs_features: List[Vector] = []
        self._obs_class_ids: List[int] = []
        self.keep_observations: bool = keep_observations

        # initialize predictions
        # these are updated by the predict method
        self._pred_boxes: List[Box] = []
        self._pred_scores: List[float] = []
        self._pred_features: List[Vector] = []
        self._pred_class_ids: List[int] = []
        self.keep_predictions: bool = keep_predictions

        logger.debug(f"creating new tracker {self.id}")

    @property
    def box(self) -> Box:
        return self._pred_boxes[-1]

    @property
    def score(self) -> float:
        return self._pred_scores[-1]

    @property
    def feature(self) -> Vector:
        return self._pred_features[-1]

    @property
    def class_id(self) -> int:
        return self._pred_class_ids[-1]

    @box.setter
    def box(self, box: Box) -> None:
        raise AttributeError("box is read-only")

    @score.setter
    def score(self, score: float) -> None:
        raise AttributeError("score is read-only")

    @feature.setter
    def feature(self, feature: Vector) -> None:
        raise AttributeError("feature is read-only")

    @class_id.setter
    def class_id(self, class_id: int) -> None:
        raise AttributeError("class_id is read-only")

    def _update_observation(self, detection: Union[Detection, None]) -> None:

        box = detection.box if detection is not None else None
        score = detection.score if detection is not None else None
        feature = detection.feature if detection is not None else None
        class_id = detection.class_id if detection is not None else None

        if self.keep_observations:
            self._obs_boxes.append(box)
            self._obs_scores.append(score)
            self._obs_features.append(feature)
            self._obs_class_ids.append(class_id)
        else:
            self._obs_boxes = [box]
            self._obs_scores = [score]
            self._obs_features = [feature]
            self._obs_class_ids = [class_id]

    def update(
        self,
        detection: Union[Detection, None],
        no_predict: bool = False,
    ) -> None:
        """update the tracker with a new detection and predict the next state.

        Args:
            detection (Detection): detection to update the tracker with
            no_predict (bool, optional): whether to skip prediction. Defaults to False.

        Note:
            If there is no detection (i.e. detection is None), the tracker will still predict the next state.
        """

        self.steps_alive += 1

        if detection is not None:
            self.steps_positive += 1
            self.staleness = 0.0
            self._update_observation(detection)
        else:
            self.staleness += 1
            self._update_observation(None)

        if not no_predict:
            self.predict()

    def _predict(self) -> None:
        """predict the next state of the tracker"""
        # find the most recent observation
        for i in range(1, len(self._obs_boxes) + 1):
            if self._obs_boxes[-i] is not None:
                box = self._obs_boxes[-i]
                score = self._obs_scores[-i]
                feature = self._obs_features[-i]
                class_id = self._obs_class_ids[-i]
                break
        else:
            raise ValueError("no observation found")
        return box, score, feature, class_id

    def predict(self) -> None:
        box, score, feature, class_id = self._predict()

        if self.keep_predictions:
            self._pred_boxes.append(box)
            self._pred_scores.append(score)
            self._pred_features.append(feature)
            self._pred_class_ids.append(class_id)
        else:
            self._pred_boxes = [box]
            self._pred_scores = [score]
            self._pred_features = [feature]
            self._pred_class_ids = [class_id]

    def is_stale(self) -> bool:
        """check if the tracker is stale. e.g. `self.staleness > self.max_staleness`."""
        return self.staleness > self.max_staleness

    def to_bbdf(self) -> BBoxDataFrame:
        """Convert the tracker predictions to a BBoxDataFrame.

        Returns:
            BBoxDataFrame: BBoxDataFrame of the tracker
        """

        if len(self._pred_boxes) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "frame": range(len(self._pred_boxes)),
                "id": self.id,
                "box": self._pred_boxes,
                "score": self._pred_scores,
                "feature": self._pred_features,
                "class_id": self._pred_class_ids,
            }
        )

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

        bbdf = BBoxDataFrame(box_df.values, index=df.index, columns=idx)
        print(bbdf)
        return bbdf

    def __repr__(self) -> str:
        return f"(box: {str(self.box)}, score: {self.score}, class_id: {self.class_id}, staleness: {self.staleness:.2f})"


class ExponentialMovingAverage:
    def __init__(self, gamma: float = 0.5):
        self.gamma = gamma
        self._value = None

    def update(self, value: Union[float, np.ndarray]) -> None:
        if self._value is None:
            self._value = value
        else:
            self._value = self.gamma * self._value + (1 - self.gamma) * value
        return self._value

    def predict(self) -> None:
        return self._value

    @property
    def value(self) -> Union[float, np.ndarray]:
        return self._value


class SimpleTracker(SingleObjectTracker):
    """A simple single tracker with no motion modeling and box update using exponential moving averege"""

    def __init__(
        self,
        smooth_score_gamma: float = 0.5,
        smooth_feature_gamma: float = 0.5,
        smooth_box_gamma: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.smooth_score = ExponentialMovingAverage(smooth_score_gamma)
        self.smooth_feature = ExponentialMovingAverage(smooth_feature_gamma)
        self.smooth_box = ExponentialMovingAverage(smooth_box_gamma)

    def _predict(self) -> None:
        """predict the next state of the tracker"""

        # find the most recent observation
        for i in range(1, len(self._obs_boxes) + 1):
            if self._obs_boxes[-i] is not None:
                box = self.smooth_box.update(self._obs_boxes[-i])
                score = self.smooth_score.update(self._obs_scores[-i])
                feature = self.smooth_feature.update(self._obs_features[-i])
                class_id = self._obs_class_ids[-i]
                break
        else:
            raise ValueError("no observation found")
        return box, score, feature, class_id