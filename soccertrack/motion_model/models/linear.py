import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SingleTargetLinear(nn.Module):
    def __init__(self, obs_steps):
        """Simple linear model that predicts the next position based on the last `obs_steps`, using a constant velocity model."""
        super().__init__()
        self.obs_steps = obs_steps

    def forward(self, x):
        if x.shape[1] == 1:
            # If only one observation, just return it
            return x[:, -1]

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


if __name__ == "__main__":
    model = SingleTargetLinear(5)
    x = torch.randn(2, 20, 2)  # (batch_size, seq_len, input_dim)
    y_pred = model.roll_out(x, 5)  # (batch_size, n_steps, 2)
