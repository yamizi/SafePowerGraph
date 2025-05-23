import sys

sys.path.append(".")
from utils.pandapower.pandapower_graph import PandaPowerGraph
import pandapower as pp
import torch_geometric.transforms as T
import networkx as nx
from matplotlib import pyplot as plt
from torch_geometric.nn import to_hetero
from torch_geometric.utils import to_networkx

if __name__ == "__main__":
    device = "cpu"
    case = "case9"
    case_method = getattr(pp.networks, case)
    network = case_method()
    transforms = [T.ToUndirected(merge=True), T.ToDevice(device)]

    graph_hetero = PandaPowerGraph(network, include_res=True, opf_as_y=True, preprocess='metapath2vec',
                                   transform=T.Compose(transforms), scale=False, hetero=True)

    homogene = graph_hetero.data.to_homogeneous()
    colors = ["red", "green", "blue", "purple", "orange"]
    node_types = ["Bus", "Load", "External Grid", "Generator", "Line"]
    color_map = []
    g = to_networkx(homogene)
    for node in g:
        color_map.append(colors[homogene.node_type[node]])
    nx.draw(g, node_color=color_map, with_labels=True, pos=nx.planar_layout(g))
    plt.show()

    graph_homo = PandaPowerGraph(network, include_res=True, opf_as_y=True, preprocess='metapath2vec',
                                 transform=T.Compose(transforms), scale=False, hetero=False)

    g_homo = to_networkx(graph_homo.data)
    nx.draw(g_homo, with_labels=True, pos=nx.planar_layout(g_homo))
    plt.show()
