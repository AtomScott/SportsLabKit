import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class SingleTargetLSTM(nn.Module):
    def __init__(self, hidden_dim, n_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(2, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc0 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn0 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc1 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 4)
        self.fc2 = nn.Linear(hidden_dim // 4, 2)

    def forward(self, x, return_states=False, h0=None, c0=None):
        if h0 is None:
            h0 = torch.zeros(self.n_layers, x.shape[0], self.lstm.hidden_size)
        if c0 is None:
            c0 = torch.zeros(self.n_layers, x.shape[0], self.lstm.hidden_size)

        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # return the last prediction
        last_lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # predict the displacement from the last position
        _y_disp = F.relu(self.bn0(self.fc0(last_lstm_out)))
        _y_disp = F.relu(self.bn1(self.fc1(_y_disp)))
        y_disp = self.fc2(_y_disp)  # (batch_size, 2)

        # predict the next position
        y_pred = x[:, -1, :] + y_disp  # (batch_size, 1, 2)

        if return_states:
            return y_pred, (h_n, c_n)
        return y_pred

    def roll_out(self, x, n_steps, y_gt=None):
        y_pred = []
        h_n = None
        c_n = None

        for i in range(n_steps):
            y_pred_i, (h_n, c_n) = self.forward(x, return_states=True, h0=h_n, c0=c_n)
            y_pred.append(y_pred_i)

            if y_gt is not None:
                # use the ground truth position as the next input
                x = y_gt[:, i : i + 1, :]
            else:
                # use the predicted position as the next input
                x = y_pred_i.unsqueeze(1)  # (batch_size, 1, 2)

        return torch.stack(y_pred, dim=1)


class MultiTargetLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True
        )
        self.fc0 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc1 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc2 = nn.Linear(hidden_dim // 4, input_dim)

    def forward(self, x, return_states=False, h0=None, c0=None):
        if h0 is None:
            h0 = torch.zeros(self.n_layers, x.shape[0], self.lstm.hidden_size)
        if c0 is None:
            c0 = torch.zeros(self.n_layers, x.shape[0], self.lstm.hidden_size)

        # B: batch size, L: sequence length, N: number of agents, D: dimension
        num_agents = x.shape[2]
        x = rearrange(x, "B L N D -> B L (N D)")

        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # return the last prediction
        last_lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # predict the displacement from the last position
        _y_disp = self.fc0(last_lstm_out)
        _y_disp = F.relu(_y_disp)
        _y_disp = self.fc1(_y_disp)
        _y_disp = F.relu(_y_disp)
        y_disp = self.fc2(_y_disp)  # (batch_size, 2)

        # predict the next position
        y_pred = x[:, -1, :] + y_disp  # (batch_size, 1, 2)

        # B: batch size, L: sequence length, N: number of agents, D: dimension
        y_pred = rearrange(y_pred, "B (N D) -> B N D", N=num_agents)

        if return_states:
            return y_pred, (h_n, c_n)
        return y_pred

    def roll_out(self, x, n_steps, y_gt=None):
        y_pred = []
        h_n = None
        c_n = None

        for i in range(n_steps):
            y_pred_i, (h_n, c_n) = self.forward(x, return_states=True, h0=h_n, c0=c_n)
            y_pred.append(y_pred_i)

            if y_gt is not None:
                # use the ground truth position as the next input
                x = y_gt[:, i : i + 1, :]
            else:
                # use the predicted position as the next input
                x = y_pred_i.unsqueeze(1)  # (batch_size, 1, 2)

        return torch.stack(y_pred, dim=1)


if __name__ == "__main__":
    model = SingleTargetLSTM(32, n_layers=2, dropout=0.5)
    x = torch.randn(2, 20, 2)  # (batch_size, seq_len, input_dim)
    y_pred = model.roll_out(x, 5)  # (batch_size, n_steps, 2)
    print(y_pred.shape)
