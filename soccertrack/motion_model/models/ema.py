from typing import Any, Dict, Type, Union, Tuple

import numpy as np
from soccertrack.motion_model.base import MotionModel


class ExponentialMovingAverage(MotionModel):
    """
    Exponential Moving Average (EMA) motion model for object tracking.

    This class implements an EMA-based motion model for object tracking.
    It can be used both in a stateful and a procedural manner.

    Attributes:
        gamma (float): The weight for the exponential moving average calculation.
        _value (Union[float, np.ndarray, None]): The internal state of the motion model.
    """

    hparam_search_space = {"gamma": {"type": "float", "low": 0.0, "high": 1.0}}
    required_observation_types = ["box"]
    required_state_types = ["EMA_t"]

    def __init__(self, gamma: float = 0.5):
        """
        Initialize the ExponentialMovingAverage motion model.

        Args:
            gamma (float): The weight for the exponential moving average calculation. Default is 0.5.
        """
        super().__init__()
        self.gamma = gamma
        self._value = None

    def predict(
        self,
        observations: Union[float, np.ndarray],
        states: Union[float, np.ndarray, None],
    ) -> Tuple[Union[float, np.ndarray, None], Union[float, np.ndarray]]:
        boxes = np.array(observations.get("box", None))
        EMA_t = states["EMA_t"]

        if EMA_t is None:
            # compute EMA_t from all boxes
            for box in boxes:
                EMA_t = box if EMA_t is None else self.gamma * EMA_t + (1 - self.gamma) * box
        else:
            box = boxes[-1:].squeeze()
            EMA_t = self.gamma * EMA_t + (1 - self.gamma) * box
        new_states = {"EMA_t": EMA_t}
        return EMA_t, new_states
