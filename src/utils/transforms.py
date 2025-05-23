from torch_geometric.transforms import BaseTransform
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData


class ToTransformerInput(BaseTransform):
    def __init__(self, output_dim=64):
        super().__init__()
        self.output_dim = output_dim
        self.projection_layers = nn.ModuleDict()

    def __call__(self, data: HeteroData) -> HeteroData:
        # Project node features to common dimension
        node_projections = {}
        for node_type in data.node_types:
            feat = data[node_type].x
            input_dim = feat.size(-1)

            if node_type not in self.projection_layers:
                self.projection_layers[node_type] = nn.Linear(input_dim, self.output_dim, bias=False)

            node_projections[node_type] = self.projection_layers[node_type](feat)

        # Project edge features (if exist)
        edge_projections = {}
        for edge_type in data.edge_types:
            if hasattr(data[edge_type], 'edge_attr'):
                feat = data[edge_type].edge_attr
                input_dim = feat.size(-1)

                if edge_type not in self.projection_layers:
                    self.projection_layers[str(edge_type)] = nn.Linear(input_dim, self.output_dim, bias=False)

                edge_projections[edge_type] = self.projection_layers[str(edge_type)](feat)
            else:
                # Create zero-initialized edge features if none exist
                edge_projections[edge_type] = torch.zeros(data[edge_type].edge_index.size(1), self.output_dim)

        # Create unified sequences
        node_sequences = []
        for node_type, proj in node_projections.items():
            # Add positional encoding (index-based)
            pos_enc = torch.arange(proj.size(0)).unsqueeze(-1).float()
            # Add type embedding
            type_emb = torch.full((proj.size(0), 1), ord(node_type[0]) % 100)  # Simple hash-based
            node_sequences.append(torch.cat([proj, pos_enc, type_emb], dim=-1))

        edge_sequences = []
        for edge_type, proj in edge_projections.items():
            pos_enc = torch.arange(proj.size(0)).unsqueeze(-1).float()
            type_emb = torch.full((proj.size(0), 1), hash(edge_type) % 100)
            edge_sequences.append(torch.cat([proj, pos_enc, type_emb], dim=-1))

        # Combine all sequences

        combined_sequence = torch.cat([
            torch.cat(node_sequences, dim=0),
            torch.cat(edge_sequences, dim=0)
        ], dim=0)

        data.transformer_input = combined_sequence
        # Store original structure metadata
        data.metadata = {
            'num_nodes': sum(len(p) for p in node_projections.values()),
            'num_edges': sum(len(p) for p in edge_projections.values()),
            'node_types': list(node_projections.keys()),
            'edge_types': list(edge_projections.keys())
        }

        data.attention_mask = torch.ones(len(combined_sequence), dtype=torch.bool)

        return data


def collate_fn(batch):
    # Pad sequences to max length in batch
    sequences = [data.transformer_input for data in batch]
    targets = [data.y for data in batch] if hasattr(batch[0], 'y') else None

    seq_lens = [s.size(0) for s in sequences]
    max_len = max(seq_lens)
    feat_dim = sequences[0].size(-1)

    padded = torch.zeros(len(batch), max_len, feat_dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, (seq, length) in enumerate(zip(sequences, seq_lens)):
        padded[i, :length] = seq
        mask[i, :length] = 1

    output = {'inputs': padded, 'attention_mask': mask}
    if targets is not None:
        output['targets'] = torch.stack(targets)

    return output
