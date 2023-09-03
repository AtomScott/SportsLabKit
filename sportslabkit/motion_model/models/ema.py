from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from sportslabkit.motion_model.base import BaseMotionModel


class ExponentialMovingAverage(BaseMotionModel):
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

    def __init__(self, model_config={}, inference_config={}):
        """
        Initialize the ExponentialMovingAverage motion model.

        Args:
            gamma (float): The weight for the exponential moving average calculation. Default is 0.5.
        """
        super().__init__(model_config, inference_config)
        self._value = None

    def predict(
        self,
        observations: Union[float, np.ndarray],
        states: Union[float, np.ndarray, None],
    ) -> Tuple[Union[float, np.ndarray, None], Union[float, np.ndarray]]:
        gamma = self.model_config["gamma"]

        boxes = np.array(observations.get("box", None))
        EMA_t = states["EMA_t"]

        if EMA_t is None:
            # compute EMA_t from all boxes
            for box in boxes:
                EMA_t = box if EMA_t is None else gamma * EMA_t + (1 - gamma) * box
        else:
            box = boxes[-1:].squeeze()
            EMA_t = gamma * EMA_t + (1 - gamma) * box
        new_states = {"EMA_t": EMA_t}
        return EMA_t, new_states
