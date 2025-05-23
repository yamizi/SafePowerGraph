import os.path
import sys
import hashlib

sys.path.append(".")
import uuid
import numpy as np
import torch
import json, os
import matplotlib.pyplot as plt
from runs.cases import init_device, init_model, init_dataset, run_train_eval
import pandapower.plotting as plotting
import pandapower as pp
import seaborn
seaborn.set()

from copy import deepcopy
def run_case(training_cases=[["case9", 64, 0.7, ["cost", "load"]]], experiment=None,
             validation_case=["case9", 64, 0.7, ["cost", "load"]], plot=True,
             save_path="../output", dataset_type="y_OPF",
             scale=False, y_nodes=["gen", "ext_grid", "bus"],
             device="cuda", opf=2, use_ray=True, uniqueid="", hidden_channels=[64, 64],
             base_lr=0.1, decay_lr=0.5, cls="sage", aggr="mean",
             clamp_boundary=0, use_physical_loss=1, weighting="relative", seed=1,
             initial_epoch=0, hetero=True, model_file=None, num_workers_train=0):
    device = init_device(device, seed)
    if not uniqueid:
        uniqueid = uuid.uuid4()

    title = "analyze_lines"
    max_epochs = 0
    val_batch_size = 1
    train_batch_size = 5
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    threshold = 5


    os.makedirs(save_path, exist_ok=True)
    pickle_file = f"{save_path}/{uniqueid}_{seed}_{hetero}.pkl"
    pin_memory = device == "cpu" or num_workers_train != 0
    num_workers_train = int(num_workers_train)

    all_params = {"uniqueid": uniqueid, "max_epochs": max_epochs, "dataset_type": dataset_type,
                  "scale": scale, "hetero": hetero, "opf": opf, "use_ray": use_ray,
                  "save_path": save_path, "title": title, "y_nodes": y_nodes, "plot": plot,
                  "device": device, "train_batch_size": train_batch_size,
                  "hidden_channels": hidden_channels,
                  "val_batch_size": val_batch_size, "pickle_file": pickle_file,
                  "clamp_boundary": clamp_boundary, "use_physical_loss": use_physical_loss,
                  "base_lr": base_lr, "cls": cls, "aggr": aggr,
                  "weighting": weighting, "losses": "mse+l1", "seed": seed,
                  "initial_epoch": initial_epoch, "num_workers_train": num_workers_train,
                  "pin_memory": pin_memory}

    all_param_hash = hashlib.md5(json.dumps(all_params).encode()).hexdigest()
    model_file = f"{save_path}/model_{uniqueid}_{seed}_{all_param_hash}.pt" if model_file is None else model_file
    figure_output_folder = f"{save_path}/figures_seed{seed}_{all_param_hash}/"
    os.makedirs(figure_output_folder, exist_ok=True)

    if experiment is not None:
        experiment.log_parameters(
            {**all_params, "model_file": model_file, "figure_output_folder": figure_output_folder})

    train_case_name, val_case_name, nb_graph, mutation_rate, mutations, train_graphs, train_networks, val_graphs, valid_networks = init_dataset(
        training_cases, validation_case, all_params, experiment, filter_dataset=False)

    graph_y = val_graphs[0]

    val_case_name, nb_graphs, mutation_rate, mutations = validation_case
    mutation = mutations[0].replace(":", "_")

    training_params = {"decay_lr": decay_lr, "base_lr": base_lr, "cls": cls, "hidden_channels": hidden_channels,
                       "aggr": aggr,
                       "train_case_name": train_case_name, "val_case_name": val_case_name}
    if experiment is not None:
        experiment.log_parameters(training_params)

    model = init_model(graph_y, all_params, aggr, hidden_channels, cls, model_file)

    model, (output, losses) = run_train_eval(model, train_graphs, train_networks, val_graphs, valid_networks,
                                             all_params, training_params, experiment)
    grid_metrics = []

    cmap_line = [(20, "green"), (50, "yellow"), (100, "orange"), (120, "red")]
    cmap_load=  [(0.9, "blue"), (0.95, "green"), (1.05, "orange"), (1.1, "red")]

    if plot:
        net = valid_networks.get("original")
        pp.runopp(net)
        fig, ax = plt.subplots()
        cmap_list = cmap_line
        cmap, norm = plotting.cmap_continuous(cmap_list)
        lc = plotting.create_line_collection(net, net.line.index, zorder=1, cmap=cmap, norm=norm, linewidths=2)
        cmap_list = cmap_load
        cmap, norm = plotting.cmap_continuous(cmap_list)
        bc = plotting.create_bus_collection(net, net.bus.index, size=0.1, zorder=2, cmap=cmap, norm=norm)
        plotting.draw_collections([lc, bc], figsize=(8, 6), ax=ax)

        figure_path = os.path.join(figure_output_folder, f'grid_{val_case_name}.png')
        ax.figure.savefig(figure_path)
        ax.figure.savefig(os.path.join(figure_output_folder, f'grid_{val_case_name}.pdf'))
        if experiment is not None:
            experiment.log_image(figure_path, name=f'grid_{val_case_name}.png', overwrite=False, step=0)

        plt.close(fig)

    for i, net in enumerate(valid_networks.get("mutants")):
        dropped_lines = net.line[net.line["in_service"] == False]
        dropped_line = net.line[net.line["in_service"] == False].index.values[0] if len(dropped_lines)>0 else ""
        grid_metric = {"dropped_line": dropped_line, "weighted": losses[i].get("weighted").mean(),
                       "cost": losses[i].get("cost").mean()
            , "physical": losses[i].get("physical").mean(), "boundary": np.hstack(losses[i].get("boundary")).mean()}
        grid_metrics.append(grid_metric)
        pp.drop_lines(net, net.line[net.line["in_service"] == False].index.values)
        # plot.plotly.pf_res_plotly(net)

        oracle_net = deepcopy(net)
        gen_p = output[i].get("gen")[:,0]*net.sn_mva
        gen_q = output[i].get("gen")[:, 1] * net.sn_mva

        ext_p = output[i].get("ext_grid")[:, 0] * net.sn_mva
        ext_q = output[i].get("ext_grid")[:, 1] * net.sn_mva

        net.gen.min_p_mw = gen_p - threshold
        net.gen.max_p_mw = gen_p + threshold
        net.res_gen.p_mw = gen_p
        net.gen.p_mw = gen_p
        net.gen.min_q_mvar = gen_q - threshold
        net.gen.max_q_mvar = gen_q + threshold
        net.res_gen.q_mvar = gen_q
        net.gen.q_mvar = gen_q

        net.ext_grid.min_p_mw = ext_p - threshold
        net.ext_grid.max_p_mw = ext_p + threshold
        net.res_ext_grid.p_mw = ext_p
        net.ext_grid.p_mw = ext_p
        net.ext_grid.min_q_mvar = ext_q - threshold
        net.ext_grid.max_q_mvar = ext_q + threshold
        net.res_ext_grid.q_mvar = ext_q
        net.ext_grid.q_mvar = ext_q

        vm = output[i].get("bus")[:,-2]
        net.bus.max_vm_pu = vm + threshold
        net.bus.min_vm_pu = vm - threshold
        net.res_bus.vm_pu = vm
        net.bus.vm_pu = vm

        #pp.runopp(net, init="results")


        if plot:

            fig, ax = plt.subplots()
            cmap_list = cmap_line
            cmap, norm = plotting.cmap_continuous(cmap_list)
            lc = plotting.create_line_collection(oracle_net, oracle_net.line.index, zorder=1, cmap=cmap, norm=norm, linewidths=2)
            cmap_list = cmap_load
            cmap, norm = plotting.cmap_continuous(cmap_list)
            bc = plotting.create_bus_collection(oracle_net, oracle_net.bus.index, size=0.1, zorder=2, cmap=cmap, norm=norm)
            plotting.draw_collections([lc, bc], figsize=(8, 6), ax=ax)

            figure_path_oracle = os.path.join(figure_output_folder, f'grid_{i}_{mutation}_{dropped_line}_o.png')
            ax.figure.savefig(figure_path_oracle)
            ax.figure.savefig(os.path.join(figure_output_folder, f'grid_{i}_{mutation}_{dropped_line}_o.pdf'))

            fig, ax = plt.subplots()
            cmap_list = cmap_line
            cmap, norm = plotting.cmap_continuous(cmap_list)
            lc = plotting.create_line_collection(net, net.line.index, zorder=1, cmap=cmap, norm=norm, linewidths=2)
            cmap_list = cmap_load
            cmap, norm = plotting.cmap_continuous(cmap_list)
            bc = plotting.create_bus_collection(net, net.bus.index, size=0.1, zorder=2, cmap=cmap, norm=norm)
            plotting.draw_collections([lc, bc], figsize=(8, 6), ax=ax)

            figure_path = os.path.join(figure_output_folder, f'grid_{i}_{mutation}_{dropped_line}.png')
            ax.figure.savefig(figure_path)
            ax.figure.savefig(os.path.join(figure_output_folder, f'grid_{i}_{mutation}_{dropped_line}.pdf'))
            if experiment is not None:
                experiment.log_image(figure_path_oracle, name=f'oracle_{mutation}_{dropped_line}.png', overwrite=False, step=i)
                experiment.log_image(figure_path, name=f'gnn_{mutation}_{dropped_line}.png', overwrite=False, step=i)

            plt.close(fig)

    print("saved figures to", figure_output_folder)
    if experiment is not None:
        [experiment.log_metrics(e, prefix="graph_", step=i) for (i, e) in
         enumerate(grid_metrics)]

    return model, train_graphs, train_networks, val_graphs, valid_networks
