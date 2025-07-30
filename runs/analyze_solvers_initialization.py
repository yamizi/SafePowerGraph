


import os.path
import sys
import hashlib

sys.path.append(".")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uuid
import pandas as pd
import numpy as np
from copy import deepcopy
import json, os
from runs.cases import init_dataset, init_device
import pandapower as pp
import matplotlib.pyplot as plt
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

sys.path.append('/home/bizon/Projects/Max/output/matpower')


def run_case(experiment=None,
             validation_case=["case9", 64, 0.7, ["cost", "load"]],
             plot=True, save_path="../output", dataset_type="y_OPF",
             scale=False, y_nodes=["gen", "ext_grid", "bus"],
             device="cuda", opf="pandapower_opf", use_ray=False, uniqueid="", seed=4,
             initial_epoch=0, hidden_channels=[64, 64],
             base_lr=0.1, decay_lr=0.5, cls="sage", aggr="mean",
             clamp_boundary=0, use_physical_loss=1, weighting="relative", 
             hetero=True, num_workers_train=0, model_file="", num_initial_states=10):


    title = "analyze_intialization"
    training_cases = [[validation_case[0], 1, 0.7, validation_case[3]]]
    mutation = validation_case[3][0].replace(":","_")
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
    
    all_gen = []
    all_ext_grid = []
    all_bus = []

    for i, net in enumerate(valid_networks.get("mutants")):

        for run in range(num_initial_states):
            network = deepcopy(net)# Run multiple initializations
            try:
                configure_initial_set_with_randomization(network)
                pp.runopp(network, init='results')
            except pp.optimal_powerflow.OPFNotConverged as e:
                print(i, f": error opf with result init, run {run}", e)
                continue

            res_gen = network.res_gen[["p_mw","q_mvar"]] /network.sn_mva
            res_ext_grid = network.res_ext_grid[["p_mw","q_mvar"]] /network.sn_mva
            res_bus = network.res_bus[["vm_pu", "va_degree"]]
            res_bus["va_degree"] = np.deg2rad(res_bus["va_degree"])

            res_gen["run"] = run
            res_ext_grid["run"] = run
            res_bus["run"] = run

            res_gen["i"] = i
            res_ext_grid["i"] = i
            res_bus["i"] = i

            all_gen.append(res_gen)
            all_ext_grid.append(res_ext_grid)
            all_bus.append(res_bus)


    all_gen_df = pd.concat(all_gen).reset_index()
    all_ext_grid_df = pd.concat(all_ext_grid).reset_index()
    all_bus_df = pd.concat(all_bus).reset_index()

    # we use groupby(columns) instead of mean(columns) = same thing
    all_gen_std = all_gen_df.drop(columns=["run"]).groupby(["index","i"]).std()
    all_gen_mean = all_gen_df.drop(columns=["run"]).groupby(["index", "i"]).mean()

    all_ext_std = all_ext_grid_df.drop(columns=["run"]).groupby(["index", "i"]).std()
    all_ext_mean = all_ext_grid_df.drop(columns=["run"]).groupby(["index", "i"]).mean()


    all_bus_std = all_bus_df.drop(columns=["run"]).groupby(["index","i"]).std()
    all_bus_mean = all_bus_df.drop(columns=["run"]).groupby(["index", "i"]).mean()


    all_vals = {"all_gen_std":all_gen_std,"all_gen_mean":all_gen_mean,"all_ext_std":all_ext_std,"all_ext_mean":all_ext_mean,"all_bus_std":all_bus_std,"all_bus_mean":all_bus_mean}
    figure_path = ""
    for (k,v) in all_vals.items():
        fig, ax = plt.subplots()
        v.plot()
        plt.tight_layout()

        extension = "png"
        figure_path = os.path.join(figure_output_folder, f"{k}_{val_case_name}_{mutation}.{extension}")
        plt.savefig(figure_path)
        plt.savefig(os.path.join(figure_output_folder, f"{k}_{val_case_name}_{mutation}.pdf"))
        experiment.log_image(figure_path, name=f"{k}{val_case_name}_{mutation}.{extension}", overwrite=False, step=0)
        plt.close(fig)

    if experiment is not None:
        experiment.log_parameters({**all_params, "figure_path": figure_path})
        experiment.log_dataframe_profile(all_gen_df, "all_gen")
        experiment.log_dataframe_profile(all_ext_grid_df, "all_ext_grid")
        experiment.log_dataframe_profile(all_bus_df, "all_bus")

        experiment.log_dataframe_profile(all_gen_mean, "all_gen_mean")
        experiment.log_dataframe_profile(all_gen_std, "all_gen_std")
        experiment.log_dataframe_profile(all_ext_std, "all_ext_std")
        experiment.log_dataframe_profile(all_ext_mean, "all_ext_mean")
        experiment.log_dataframe_profile(all_bus_std, "all_bus_std")
        experiment.log_dataframe_profile(all_bus_mean, "all_bus_mean")


    return val_graphs, valid_networks


def configure_initial_set_with_randomization(network):
    if hasattr(network, 'res_gen') and hasattr(network, 'res_ext_grid') and hasattr(network, 'res_bus'):
        min_gen_p = network.res_gen['p_mw'].min()
        max_gen_p = network.res_gen['p_mw'].max()
        min_gen_q = network.res_gen['q_mvar'].min()
        max_gen_q = network.res_gen['q_mvar'].max()
        min_ext_grid_p = network.res_ext_grid['p_mw'].min()
        max_ext_grid_p = network.res_ext_grid['p_mw'].max()
        min_ext_grid_q = network.res_ext_grid['q_mvar'].min()
        max_ext_grid_q = network.res_ext_grid['q_mvar'].max()
        min_bus_vm = network.res_bus['vm_pu'].min()
        max_bus_vm = network.res_bus['vm_pu'].max()
        bus_va_noise = np.random.uniform(0, 360, size=network.res_bus['va_degree'].shape)  

        gen_p_noise = np.random.uniform(min_gen_p, max_gen_p, size=network.res_gen['p_mw'].shape)
        gen_q_noise = np.random.uniform(min_gen_q, max_gen_q, size=network.res_gen['q_mvar'].shape)
        ext_grid_p_noise = np.random.uniform(min_ext_grid_p, max_ext_grid_p, size=network.res_ext_grid['p_mw'].shape)
        ext_grid_q_noise = np.random.uniform(min_ext_grid_q, max_ext_grid_q, size=network.res_ext_grid['q_mvar'].shape)
        bus_vm_noise = np.random.uniform(min_bus_vm, max_bus_vm, size=network.res_bus['vm_pu'].shape)

        network.bus['va_degree'] = network.res_bus['va_degree'] + bus_va_noise
        network.gen['p_mw'] = gen_p_noise
        network.gen['q_mvar'] = gen_q_noise
        network.ext_grid['p_mw'] = ext_grid_p_noise
        network.ext_grid['q_mvar'] = ext_grid_q_noise
        network.bus['vm_pu'] = bus_vm_noise

        #print("Using randomized initial set based on random initialization")
    else:
        raise ValueError("No previous OPF results found for 'result' initial set.")