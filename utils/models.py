import torch
from torch_geometric.nn import SAGEConv, Linear, GraphConv, GATConv
from torch.nn import Linear as Linear2d
from collections import OrderedDict

CLS_MAP = {"sage": (SAGEConv, {"in_channels": (-1, -1)},0),
           "gcn": (GraphConv, {"in_channels": -1},1), "gat": (GATConv, {
        "in_channels": (-1, -1), "add_self_loops": False},0)}


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, initial_channels=None, aggr="mean", cls="sage"):
        # aggr can be mean, max or lstm
        super().__init__()

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]

        else:
            hidden_channels = [int(e) for e in hidden_channels]
        if initial_channels is None:
            initial_channels = hidden_channels[0] * 2

        nb_hidden_layers = len(hidden_channels)
        CLS, cls_params, cls_weights = CLS_MAP[cls]
        self.cls_weights = cls_weights

        params = {"out_channels": initial_channels, "aggr": aggr, **cls_params}
        # print("1 graph ", cls, params)
        self.first_conv = CLS(**params)
        self.convs = torch.nn.ModuleDict(
            OrderedDict([(f"conv{i}", CLS(out_channels=hidden_channels[i], aggr=aggr, **cls_params)) for i in
                         range(nb_hidden_layers)]))
        # print(self.convs)
        self.linear = Linear(int(hidden_channels[-1]), int(hidden_channels[-1]))
        self.linear2 = Linear(int(hidden_channels[-1]), out_channels)

    def forward(self, x, edge_index=None, edge_weight=None):

        x = self.first_conv(x, edge_index)
        x = x.relu()
        for (k, v) in self.convs.items():
            x = v(x, edge_index,edge_weight) if self.cls_weights else v(x, edge_index)
            x = x.relu()

        x = self.linear(x)
        x = self.linear2(x)
        return x


class FCNN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = Linear2d(input_channels, hidden_channels)
        self.conv2 = Linear2d(hidden_channels, hidden_channels)
        self.linear = Linear2d(hidden_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        x = x.relu()
        x = self.linear(x)
        return x
        return torch.tanh(x)
