from typing import Any

import numpy as np
from filterpy.kalman import predict, update
from numpy import ndarray

from sportslabkit.motion_model.base import BaseMotionModel


class KalmanFilter(BaseMotionModel):
    hparam_search_space: dict[str, dict[str, Any]] = {
        "dt": {"type": "categorical", "values": [10, 2, 1, 1 / 30, 1 / 60, 1 / 120]},
        "process_noise": {"type": "logfloat", "low": 1e-6, "high": 1e2},
        "measurement_noise": {"type": "logfloat", "low": 1e-3, "high": 1e2},
        "confidence_scaler": {"type": "logfloat", "low": 1e-3, "high": 100},
    }
    required_observation_types: list[str] = ["box", "score"]
    required_state_types: list[str] = ["x", "P", "F", "H", "R", "Q"]

    def __init__(
        self,
        dt: float = 1 / 30,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-3,
        confidence_scaler: float = 1.0,
    ):
        super().__init__()
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.confidence_scaler = confidence_scaler

    def get_initial_kalman_filter_states(self, box: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "x": np.array([box[0], box[1], box[2], box[3], 0, 0, 0, 0]),
            "P": np.eye(8),
            "F": self._initialize_state_transition_matrix(),
            "H": self._initialize_measurement_function(),
            "R": self._initialize_measurement_noise_covariance(),
            "Q": self._initialize_process_noise_covariance(),
        }

    def _initialize_state_transition_matrix(self) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0, 0, self.dt, 0, 0, 0],
                [0, 1, 0, 0, 0, self.dt, 0, 0],
                [0, 0, 1, 0, 0, 0, self.dt, 0],
                [0, 0, 0, 1, 0, 0, 0, self.dt],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

    def _initialize_measurement_function(self) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

    def _initialize_measurement_noise_covariance(self, confidence: float = 1) -> np.ndarray:
        # Scale measurement noise based on confidence
        # confidence has a negative correlation with measurement noise
        scale_factor = 1 / (confidence * self.confidence_scaler)
        return np.eye(4) * self.measurement_noise * scale_factor

    def _initialize_process_noise_covariance(self) -> np.ndarray:
        q = np.array(
            [
                [self.dt**4 / 4, 0, 0, 0, self.dt**3 / 2, 0, 0, 0],
                [0, self.dt**4 / 4, 0, 0, 0, self.dt**3 / 2, 0, 0],
                [0, 0, self.dt**4 / 4, 0, 0, 0, self.dt**3 / 2, 0],
                [0, 0, 0, self.dt**4 / 4, 0, 0, 0, self.dt**3 / 2],
                [self.dt**3 / 2, 0, 0, 0, self.dt**2, 0, 0, 0],
                [0, self.dt**3 / 2, 0, 0, 0, self.dt**2, 0, 0],
                [0, 0, self.dt**3 / 2, 0, 0, 0, self.dt**2, 0],
                [0, 0, 0, self.dt**3 / 2, 0, 0, 0, self.dt**2],
            ]
        )
        return q * self.process_noise

    def predict(
        self,
        observations: dict[str, [float | ndarray]],
        states: dict[str, float | ndarray] = None,
    ) -> tuple[ndarray, dict[str, float | ndarray]]:
        boxes = np.array(observations.get("box", None))
        scores = np.array(observations.get("score", 1))

        new_states = states.copy() if states else {}

        if new_states.get("x") is None:
            # Initialize Kalman filter states
            new_states.update(self.get_initial_kalman_filter_states(boxes[-1]))

            # Compute the state based on all the observations
            for box, score in zip(boxes, scores):
                new_states["x"], new_states["P"] = predict(
                    x=new_states["x"],
                    P=new_states["P"],
                    F=new_states["F"],
                    Q=new_states["Q"],
                )

                # Update measurement noise covariance matrix (R) based on the confidence score
                new_states["R"] = self._initialize_measurement_noise_covariance(score)

                new_states["x"], new_states["P"] = update(
                    new_states["x"],
                    new_states["P"],
                    box,
                    new_states["R"],
                    new_states["H"],
                )

        else:
            new_states["x"], new_states["P"] = predict(
                states["x"],
                states["P"],
                states["F"],
                states["Q"],
            )

            # Update measurement noise covariance matrix (R) based on the confidence score
            new_states["R"] = self._initialize_measurement_noise_covariance(scores[-1])

            new_states["x"], new_states["P"] = update(
                new_states["x"],
                new_states["P"],
                boxes[-1],
                new_states["R"],
                states["H"],
            )
        pred = new_states["x"][:4]
        return pred, new_states
