import torch
from torch import nn
from torch.nn import functional as F


class SingleTargetMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_prob=0.5):
        super().__init__()

        self.hidden_layers = nn.ModuleList()
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(in_dim, hidden_dim))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_dim))
            self.hidden_layers.append(nn.LeakyReLU())
            self.hidden_layers.append(nn.Dropout(p=dropout_prob))
            in_dim = hidden_dim

        self.output_layer = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        out = x
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.output_layer(out)

        return out

    def roll_out(self, x, n_steps):
        y_pred = []

        for i in range(n_steps):
            y_pred_i = self.forward(x)
            y_pred.append(y_pred_i)
            x = torch.cat([x[:, 1:, :], y_pred_i.unsqueeze(1)], dim=1)
        return torch.stack(y_pred, dim=1)


if __name__ == "__main__":
    model = SingleTargetLSTM(32, n_layers=2, dropout=0.5)
    x = torch.randn(2, 20, 2)  # (batch_size, seq_len, input_dim)
    y_pred = model.roll_out(x, 5)  # (batch_size, n_steps, 2)
    print(y_pred.shape)
