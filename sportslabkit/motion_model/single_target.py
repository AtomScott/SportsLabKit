import torch

from sportslabkit.motion_model.base import BaseMotionModel, BaseMotionModule
from sportslabkit.motion_model.models import LSTM, Linear


class SingleTargetLinear(BaseMotionModel):
    required_observation_types = ["pt"]
    required_state_types = []

    def __init__(self, obs_steps=92, roll_out_steps=144):
        super().__init__()
        self.module = BaseMotionModule(model=Linear(obs_steps), roll_out_steps=roll_out_steps)

    def predict(self, observations, states):
        obs = torch.tensor([observations['pt']], dtype=torch.float32)
        y = self.module(obs)

        return y, states

class SingleTargetLSTM(BaseMotionModel):
    required_observation_types = ["pt"]
    required_state_types = []

    def __init__(self, hidden_dim=64, n_layers=2, dropout=0.1, obs_steps=92, roll_out_steps=144):
        super().__init__()
        model = LSTM(hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
        self.module = BaseMotionModule(model=model, roll_out_steps=roll_out_steps)

    def predict(self, observations, states):
        obs = torch.tensor([observations['pt']], dtype=torch.float32)
        with torch.no_grad():
            self.module.eval()
            y = self.module(obs)
        return y, states
