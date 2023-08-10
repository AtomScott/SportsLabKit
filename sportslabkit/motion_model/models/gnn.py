import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

import torch
from torch import nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GENConv, aggr, GENConv, DeepGCNLayer
from torch.nn import LayerNorm, Linear, ReLU
from torch_geometric.data import Data, Batch


def get_norm_layer(norm_method, dim):
    if norm_method == "layer":
        return pyg_nn.LayerNorm(dim, affine=True)
    if norm_method == "graph":
        return pyg_nn.GraphNorm(dim)
    if norm_method == "batch":
        return pyg_nn.BatchNorm(dim, affine=True)
    if norm_method == "instance":
        return pyg_nn.InstanceNorm(dim, affine=True)
    if norm_method == "none" or norm_method is None:
        return nn.Identity()


def get_aggr(arg_method):
    if arg_method == "softmax":
        return aggr.SoftmaxAggregation(learn=True)
    if arg_method == "powermean":
        return aggr.PowerMeanAggregation(learn=True)
    return aggr.MultiAggregation(arg_method)


class GCNEncoder(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        dropout=0.0,
        norm=None,
        local_aggr="softmax",
        local_norm="layer",
        num_layers=2,
        use_complete_graph=False,
        jk=None,
    ):
        super().__init__()

        self.node_encoder = Linear(input_channels, hidden_channels)
        self.use_complete_graph = use_complete_graph

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(
                hidden_channels,
                hidden_channels,
                aggr=local_aggr,
                t=1.0,
                learn_t=True,
                num_layers=2,
                norm=local_norm,
                jk=jk,
            )
            norm = get_norm_layer(norm, hidden_channels)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block="res+", dropout=dropout)
            self.layers.append(layer)

        self.fc1 = nn.Linear(hidden_channels, output_channels)

        # TODO: This is a hack to get the complete graph. It should be refactored in a function that caches the edge index and the graph index
        self.edge_index_cache = {}
        self.graph_index_cache = {}

    def forward(self, batch):
        """
        x is a batch of graphs with shape (batch_size, num_nodes, num_features)
        """

        # Reshape x to (num_nodes, num_features) since batches are considered as unconected graphs in torch_geometric
        batch_size = batch.shape[0]
        num_nodes = batch.shape[1]

        x = rearrange(batch, "b n f -> (b n) f")
        edge_index = self.get_edge_index(num_nodes).to(self.get_device())

        x = self.node_encoder(x)
        x = self.layers[0](x, edge_index)

        for layer in self.layers[1:]:
            x = layer(x, edge_index)

        x = self.layers[0].act(self.layers[0].norm(x))
        out = self.fc1(x)

        # Reshape x to (batch_size, num_nodes, num_features)
        out = rearrange(out, "(b n) f -> b n f", b=batch_size)
        return out

    def get_device(self):
        return next(self.parameters()).device

    def get_edge_index(self, num_nodes):
        if self.edge_index_cache.get(num_nodes) is None:
            edge_index = []
            edge_index0 = []
            edge_index1 = []
            device = self.get_device()
            for i in range(num_nodes):
                for j in range(num_nodes):
                    edge_index0.append(i)
                    edge_index1.append(j)
            edge_index.append(edge_index0)
            edge_index.append(edge_index1)

            edge_index = torch.tensor(edge_index, dtype=torch.long)

            self.edge_index_cache[num_nodes] = edge_index

        return self.edge_index_cache[num_nodes]

    def get_graph_index(self, num_nodes, batch_size):
        if self.graph_index_cache.get((num_nodes, batch_size)) is None:
            edge_index = self.get_edge_index(num_nodes)
            batch_list = []
            for _ in range(batch_size):
                batch_list.append(Data(edge_index=edge_index, num_nodes=num_nodes))

            batch = Batch.from_data_list(batch_list)
            graph_index = batch.batch

            self.graph_index_cache[(num_nodes, batch_size)] = graph_index

        return self.graph_index_cache[(num_nodes, batch_size)]


class MultiTargetGNN(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels=2,
        use_complete_graph=True,
        n_layers=3,
        dropout=0.0,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.n_layers = n_layers
        self.dropout = dropout

        self.gcn = GCNEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=hidden_channels,
            num_layers=n_layers,
            use_complete_graph=use_complete_graph,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_channels, output_channels)

    def forward(self, x):
        # B: batch size, L: sequence length, N: number of agents, D: dimension
        num_agents = x.shape[2]
        model_input = rearrange(x, "B L N D -> B N (L D)")

        out = self.gcn(model_input)

        # predict the displacement from the last position
        y_disp = self.fc(out)  # (batch_size, 2)

        # predict the next position
        y_pred = x[:, -1, :] + y_disp  # (batch_size, 1, 2)

        return y_pred

    # def forward(self, x, return_states=False, h0=None, c0=None):
    #     # B: batch size, L: sequence length, N: number of agents, D: dimension
    #     num_agents = x.shape[2]
    #     x = rearrange(x, "B L N D -> B L (N D)")

    #     lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))

    #     # return the last prediction
    #     last_lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

    #     # predict the displacement from the last position
    #     _y_disp = self.fc0(last_lstm_out)
    #     _y_disp = F.relu(_y_disp)
    #     _y_disp = self.fc1(_y_disp)
    #     _y_disp = F.relu(_y_disp)
    #     y_disp = self.fc2(_y_disp)  # (batch_size, 2)

    #     # predict the next position
    #     y_pred = x[:, -1, :] + y_disp  # (batch_size, 1, 2)

    #     # B: batch size, L: sequence length, N: number of agents, D: dimension
    #     y_pred = rearrange(y_pred, "B (N D) -> B N D", N=num_agents)

    #     if return_states:
    #         return y_pred, (h_n, c_n)
    #     return y_pred

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
