import torch
from argparse import Namespace
from torch_geometric.data import HeteroData
def hetero_to_homo(l):
    output_dim = l[0].y_dict["bus"].shape[1]
    homo_data = []
    for hetero_data in l:
        for node_type in hetero_data.node_types:
            N = hetero_data[node_type]['x'].size(0)  # Number of nodes of this type
            if 'y' not in hetero_data[node_type]:
                hetero_data[node_type]['y'] = torch.zeros(N, output_dim)
            if 'boundaries' not in hetero_data[node_type]:
                hetero_data[node_type]['boundaries'] = torch.zeros(N, 2)
            if 'output_mask' not in hetero_data[node_type]:
                hetero_data[node_type]['output_mask'] = torch.zeros(N, 6)

        homo_data.append(hetero_data.to_homogeneous(node_attrs=['x', 'y', 'boundaries', 'output_mask']))
            
    return homo_data

def homo_to_hetero(data, out_, node_types):
    features_per_node = {"bus":8,"shunt":10,"line":12, "load":8, "gen":20, "ext_grid":15,"sgen":7,"trafo":29, 
                         "trafo3w":35,"impedance":8,"xward":10}
    hetero_data = HeteroData()
    hetero_dict = {}
    out = {}
    for node_type_id, node_type_name in enumerate(node_types):
        mask = (data.node_type == node_type_id)
        if node_type_name == "line":
            nb_outputs = 4
        else:
            nb_outputs = 2

        hetero_data[node_type_name].x = data.x[mask,0:features_per_node.get(node_type_name,-1)]
        hetero_data[node_type_name].y = data.y[mask, 0:nb_outputs]
        hetero_data[node_type_name].boundaries = data.boundaries[mask]
        hetero_data[node_type_name].output_mask = data.output_mask[mask]
        out[node_type_name] = out_[mask]

    edge_index = data.edge_index
    src_types = data.node_type[edge_index[0]]
    dst_types = data.node_type[edge_index[1]]

    edge_index_dict = {}

    for src_type_id, src_type_name in enumerate(node_types):
        for dst_type_id, dst_type_name in enumerate(node_types):
            # Mask: edges where source and target node types match
            mask = (src_types == src_type_id) & (dst_types == dst_type_id)
            # Edge indices of these type pairs
            sub_edge_index = edge_index[:, mask]
            if sub_edge_index.shape[1] > 0:
                edge_index_dict[(src_type_name, "to",dst_type_name)] = sub_edge_index
                hetero_data[(src_type_name, "to", dst_type_name)].edge_index = sub_edge_index

    return hetero_data, out