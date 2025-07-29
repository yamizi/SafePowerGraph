from typing import Any, Dict, Optional

import torch
from torch import nn
from torch_geometric.nn import GPSConv

from utils.models import GNN

class GPS(GNN):
    def __init__(self,in_channels, hidden_channels, out_channels, pe_channels):
        torch.nn.Module.__init__(self)
        self.lin_pe = nn.Linear(pe_channels, hidden_channels[0])
        self.input_mlp = nn.Linear(in_channels, hidden_channels[0])

        self.layers = nn.ModuleList()
        for i in range(len(hidden_channels)):
            in_c = hidden_channels[i - 1] if i > 0 else hidden_channels[0]
            out_c = hidden_channels[i]
            # For GPSConv, channels argument corresponds to output channels
            # A conv layer inside GPSConv can be None or something like GCNConv(in_c, out_c)
            self.layers.append(
                GPSConv(
                    channels=out_c,
                    conv=None,
                    heads=4,
                    dropout=0.1,
                    act='gelu',
                    norm="batch_norm",
                    attn_type='multihead'
                )
            )

        self.output = nn.Linear(hidden_channels[-1], out_channels)

    def forward(self, data):
        # Combine node feature and positional encoding initializations
        x = self.input_mlp(data.x) + self.lin_pe(data.pe)

        # Pass through GPSConv layers sequentially
        for layer in self.layers:
            x = layer(x, data.edge_index, data.batch)

        out = self.output(x)
        return out