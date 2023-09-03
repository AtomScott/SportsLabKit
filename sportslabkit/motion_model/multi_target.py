import torch

from sportslabkit.motion_model.base import BaseMotionModel, BaseMotionModule
from sportslabkit.motion_model.models.linear import Linear


class MultiTargetLinear(BaseMotionModel):
    required_observation_types = ["pt"]
    required_state_types = []

    def __init__(self, obs_steps=92, roll_out_steps=144):
        super().__init__(is_multi_target=True)
        self.module = BaseMotionModule(model=Linear(obs_steps), roll_out_steps=roll_out_steps)

    def predict(self, all_observations, all_states):
        all_ys = []
        all_new_states = []
        for i, observations in enumerate(all_observations):
            obs = torch.tensor([observations['pt']], dtype=torch.float32)
            y = self.module(obs)
            all_ys.append(y)
            all_new_states.append(all_states[i])

        return all_ys, all_new_states

