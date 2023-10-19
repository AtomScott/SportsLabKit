from typing import Any

import numpy as np
import torch
from torch import nn

from sportslabkit.motion_model.base import BaseMotionModel


# TODO: Refactor GroupCast out of slk code
class Linear(nn.Module):
    def __init__(self, obs_steps: int):
        """Simple linear model that predicts the next position based on the last `obs_steps`, using a constant velocity model."""
        super().__init__()
        self.obs_steps = obs_steps

    def forward(self, x):
        # assume x is (batch_size, seq_len, 2)
        assert x.dim() == 3 or x.dim() == 2
        if x.dim() == 2:
            # If only one observation, add a batch dimension
            x = x.unsqueeze(0)

        if x.shape[1] == 1:
            # If only one observation, just return it
            return x

        # Estimate the velocity
        v = x[:, -self.obs_steps :].diff(dim=1).mean(dim=1)  # (batch_size, 2)
        y_pred = x[:, -1] + v  # (batch_size, 2)
        return y_pred

    def roll_out(self, x, n_steps, y_gt=None):
        y_pred = []

        for i in range(n_steps):
            y_pred_i = self.forward(x)
            y_pred.append(y_pred_i)

            if y_gt is not None:
                # use the ground truth position as the next input
                x = torch.cat([x[:, 1:, :], y_gt[:, i, :].unsqueeze(1)], dim=1)
            else:
                # use the predicted position as the next input
                x = torch.cat([x[:, 1:, :], y_pred_i.unsqueeze(1)], dim=1)

        return torch.stack(y_pred, dim=1)


class GCLinear(BaseMotionModel):
    """ """

    hparam_search_space: dict[str, dict[str, object]] = {}
    required_observation_types = ["pt"]
    required_state_types = []

    def __init__(self, obs_steps: int = 25):
        """
        Initialize the ExponentialMovingAverage motion model.

        """
        super().__init__()
        self.model = Linear(obs_steps=obs_steps)

    def predict(
        self,
        observations: dict[str, Any],
        states: dict[str, float | np.ndarray[Any, Any]] = ...,
    ) -> tuple[np.ndarray[Any, Any], dict[str, float | np.ndarray[Any, Any]]]:
        x = torch.tensor(observations.get("pt", None))
        y = self.model(x)
        return y, states
