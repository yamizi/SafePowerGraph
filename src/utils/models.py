import os
import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv, GATConv, SAGEConv, to_hetero, Linear, MLP, BatchNorm


def get_model(layers="GraphConv",data=None,smoothed=False,debug=False, device="cpu", model_ckp=None,
              log=True ,**kwargs):
    print("loading model")
    if "SAGEConv" in layers or "GraphConv" in layers or "GATConv" in layers or "TransformerConv" in layers:
        model = BaseModel(layers=layers, **kwargs)

        if data is not None:
            model = to_hetero(model, data.metadata(), debug=debug)

    elif "Transformer" in layers:
        meta = {k: data[k].x.shape for k in data.node_types}
        model = OPFTransformerModel(layers=layers, metadata=meta, **kwargs)
    else:
        raise ValueError("Invalid model type")

    if log:
        print(model)

    if model_ckp is not None:
        if os.path.exists(model_ckp):
            model.load_state_dict(torch.load(model_ckp, map_location="cpu", weights_only=False))

    model = model.to(device)
    model.eval()
    return model

class OPFTransformerModel(nn.Module):
    def __init__(self,  layers: str, metadata, **kwargs):

        super().__init__()

        layers = layers.split(":")
        feature_dim = layers[1] if len(layers) > 1 else kwargs.get("projector_dimension") + 2
        hidden_dim = layers[2] if len(layers) > 2 else kwargs.get("hidden_dim", 64)
        num_layers = layers[3] if len(layers) > 3 else kwargs.get("num_layers", 6)
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(feature_dim)
        self.num_layers = int(num_layers)
        
        self.encoder= HeteroGraphTransformer(feature_dim=self.feature_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers)  # 64 + 2
        self.regressor = OPFRegressor(hidden_dim=self.hidden_dim, nodes_metadata = metadata)

    def forward(self, data):
        transformer_data = data.transformer_input
        reshaped_data = transformer_data.reshape((len(data), -1, self.feature_dim+2))
        mask = data.attention_mask.reshape((len(data), -1))
        transformer_out = self.encoder(reshaped_data, mask)
        opf_out = self.regressor(transformer_out)
        return opf_out

class HeteroGraphTransformer(nn.Module):
    def __init__(self,
                 feature_dim: int,  # Must match transform's output_dim
                 hidden_dim: int = 64,
                 num_heads: int = 8,
                 num_layers: int = 6):
        super().__init__()

        # Split dimensions from transform output
        self.orig_feat_dim = feature_dim - 2  # Original features after projection
        self.pos_dim = 1  # Positional component
        self.type_dim = 1  # Type identifier

        # Projection layers
        self.feat_proj = nn.Linear(self.orig_feat_dim, hidden_dim)
        self.pos_proj = nn.Linear(self.pos_dim, hidden_dim)
        self.type_embed = nn.Embedding(1000, hidden_dim)  # Adjust num_types as needed

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=num_layers
        )

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor):
        # Split components [features, position, type_id]
        features = inputs[..., :self.orig_feat_dim]
        positions = inputs[..., self.orig_feat_dim:self.orig_feat_dim + self.pos_dim]
        type_ids = inputs[..., -1].long()  # Type IDs should be integers

        # Project components
        feat_emb = self.feat_proj(features)
        pos_emb = self.pos_proj(positions)
        type_emb = self.type_embed(type_ids)

        # Combine embeddings
        x = feat_emb + pos_emb + type_emb

        # Transformer processing
        padding_mask = ~attention_mask.bool()
        return self.transformer(x, src_key_padding_mask=padding_mask)


class OPFRegressor(nn.Module):
    def __init__(self, hidden_dim, nodes_metadata, output_size = 2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.bus_head = nn.Linear(hidden_dim, output_size*nodes_metadata["bus"][0]) # va, vm
        self.generator_head = nn.Linear(hidden_dim, output_size*nodes_metadata["generator"][0])  # pg, qg

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        out= {
            'bus': self.bus_head(x).reshape((-1,2)),
            'generator': self.generator_head(x).reshape((-1,2))
        }

        return out



class BaseModel(torch.nn.Module):
    def __init__(self,layers="GraphConv",hidden_dim: int = 64,out_channels=2, **kwargs):
        super().__init__()
        defaut_input = -1
        layers = layers.split(":")
        if len(layers) > 1:
            layers = layers + [hidden_dim,hidden_dim]
        if layers[0] == "GraphConv":
            cls = GraphConv
        elif layers[0] == "SAGEConv":
                cls = GraphConv
        elif layers[0] == "GATConv":
            cls = GATConv
            kwargs["heads"] = 4
            kwargs["concat"] = False
            kwargs["add_self_loops"] = False
            defaut_input = (-1, -1)
        else:
            raise ValueError("Invalid layer type")

        encoder = []
        bns = []
        for i in range(1,len(layers)-1):
            out_channel = int(layers[i])
            encoder.append(cls(in_channels=defaut_input, out_channels=out_channel, **kwargs))
            bns.append(BatchNorm(out_channel))

        ### sequential of module list
        self.encoder =  nn.ModuleList(encoder)
        self.bns = nn.ModuleList(bns)
        self.conv2 = cls(in_channels=int(layers[-1]), out_channels=out_channels, **kwargs)
        #self.mlp = MLP(in_channels=-1, hidden_channels=hidden_dim//2, out_channels=out_channels, num_layers=2)


    def forward(self, x, edge_index):
        for i, layer in enumerate(self.encoder):
            x = layer(x, edge_index)
            x = self.bns[i](x)  # + self.lin1(x)
            x = x.relu()

        x = self.conv2(x, edge_index)  #+ self.lin2(x)
        #x = x.relu()
        #x = self.mlp(x)
        return x
