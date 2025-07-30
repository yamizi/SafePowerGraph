import numpy as np
import pandas as pd
import sys

sys.path.append(".")

from utils.pandapower import build_dataset
import pandapower as pp
from torch_geometric.nn import to_hetero

from torch_geometric.loader import NeighborLoader, DataLoader
import torch
from utils.models import GNN
from utils.train import train_opf
from utils.plot import plot_losses

from pandapower.plotting import simple_plot


# simple_plot(network, plot_loads=True)

def run_case(case_name="case9", nb_graphs=64, save_path="./output",
             mutation_rate=0.7, mutations=["cost", "load"], title=""):
    graphs, networks, save_path, uniqueid = build_dataset(case_name, nbsamples=nb_graphs, save_dataframes=save_path,
                                                          mutation_rate=mutation_rate, mutations=mutations)
    print("valid graphs {}/{}".format(len(graphs), nb_graphs))
    split_index = int(len(graphs) * 3 / 4)

    if len(graphs) == 0:
        return

    graph_y = graphs[0]
    data = graph_y[0]
    model = GNN(hidden_channels=64, out_channels=graph_y.num_outputs)
    model = to_hetero(model, data.metadata(), aggr='sum')

    train_loader = DataLoader([g[0] for g in graphs[:split_index]], batch_size=5)
    val_loader = DataLoader([g[0] for g in graphs[split_index:]], batch_size=5)

    train_losses, val_losses, val_losses_gen, val_losses_ext_grid, last_out = train_opf(model, train_loader, val_loader)
    plot_losses(train_losses, val_losses, val_losses_gen, val_losses_ext_grid, case_name, title, save_path)


run_case(case_name="case9", nb_graphs=64, mutations=["load"], title="load mutation", save_path="./output/case9")
run_case(case_name="case9", nb_graphs=64, mutations=["cost"], title="cost mutation", save_path="./output/case9")
run_case(case_name="case9", nb_graphs=64, mutations=["load", "cost"], title="load & cost mutation",
         save_path="./output/case9")
# run_case(case_name="case14", nb_graphs = 64)
# run_case(case_name="case30", nb_graphs = 64)
# run_case(case_name="case118", nb_graphs = 64)
# run_case(case_name="case300", nb_graphs = 64)
# run_case(case_name="example_multivoltage", nb_graphs = 64, mutation_rate=0)
