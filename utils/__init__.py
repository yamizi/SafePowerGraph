import torch

def hetero_to_homo(l):
    output_dim = l[0].y_dict["bus"].shape[1]
    homo_data = []
    for hetero_data in l:
        for node_type in hetero_data.node_types:
            if 'y' not in hetero_data[node_type]:
                N = hetero_data[node_type]['x'].size(0)  # Number of nodes of this type
                hetero_data[node_type]['y'] = torch.zeros(N, output_dim)

        homo_data.append(hetero_data.to_homogeneous(node_attrs=['x', 'y']))
            
    return homo_data