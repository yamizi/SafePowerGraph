from typing import Any, Dict, Optional

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.nn import GraphConv
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention

from utils.models import GNN

class GPS(GNN):
    def __init__(self,hidden_channels, out_channels, embedding_channels=64, pe_dim: int=0,
                 attn_type: str="multihead"):
        torch.nn.Module.__init__(self)
        attn_kwargs = {'dropout': 0.5}
        defaut_input = -1
        channels = embedding_channels
        num_layers = len(hidden_channels)
        self.node_emb = Embedding(28, channels - pe_dim)
        self.pe_lin = Linear(20, pe_dim) if pe_dim>0 else None
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(8, channels)

        self.convs = ModuleList()
        for i in range(num_layers):
            out_channel = int(hidden_channels[i])
            conv = GraphConv(in_channels=defaut_input, out_channels=out_channel)
            conv = GPSConv(channels, conv, heads=4,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.linear = Linear(int(hidden_channels[-1]), int(hidden_channels[-1]))
        self.linear2 = Linear(int(hidden_channels[-1]), out_channels)

        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, x, edge_index, edge_attr=None,pe = None):
        self.redraw_projection.redraw_projections()
        if pe is not None and self.pe_lin is not None:
            x_pe = self.pe_norm(pe)
            x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        else:
            x = self.node_emb(x.squeeze(-1))
        edge_attr = self.edge_emb(edge_attr) if edge_attr is not None else None

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)

        x = self.linear(x)
        x = self.linear2(x)
        return x



class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1
