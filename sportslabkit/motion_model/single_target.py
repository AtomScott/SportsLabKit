# import torch
# from teamtrack.mdn_module import MDNMotionModule

# from sportslabkit.motion_model.base import BaseMotionModel, BaseMotionModule
# from sportslabkit.motion_model.models import Linear


# class SingleTargetLinear(BaseMotionModel):
#     required_observation_types = ["pt"]
#     required_state_types = []

#     def __init__(self, obs_steps=92, roll_out_steps=144):
#         super().__init__()
#         self.module = BaseMotionModule(model=Linear(obs_steps), roll_out_steps=roll_out_steps)

#     def predict(self, observations, states):
#         obs = torch.tensor([observations['pt']], dtype=torch.float32)
#         y = self.module(obs)

#         return y, states

# class SingleTargetLSTM(BaseMotionModel):
#     required_observation_types = ["pt"]
#     required_state_types = []

#     # def __init__(self, hidden_dim=64, n_layers=2, dropout=0.1, obs_steps=92, roll_out_steps=144, model=None):
#     #     super().__init__()
#         # lstm = LSTM(hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
#         # self.module = BaseMotionModule(model=lstm, roll_out_steps=roll_out_steps)

#     def __init__(self, model, device='cpu'):
#         super().__init__()
#         self.module = MDNMotionModule.load_from_checkpoint(model, map_location=torch.device(device))

#     def predict(self, observations, states):
#         obs = torch.tensor([observations['pt']], dtype=torch.float32)
#         with torch.no_grad():
#             self.module.eval()
#             obs = self.module.downsample(torch.Tensor(obs))
#             next_position = self.module.model.predict(obs)

#         # concatenate the last observation with the next position to get a trajectory to downsample
#         y = torch.cat([obs, next_position.unsqueeze(1)], dim=1)
#         y = self.module.upsample(y)[:, 1]
#         return y, states
