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

from utils.pandapower.opf_validation import get_boundary_constraints_violations
from utils.train.helpers import power_imbalance_loss

def run_case(training_cases=[["case9", 64, 0.7, ["cost", "load"]]], experiment=None,
             validation_case=["case9", 64, 0.7, ["cost", "load"]], plot=True,
             save_path="../output", dataset_type="y_OPF",
             scale=False, y_nodes=["gen", "ext_grid", "bus"],
             device="cuda", opf=2, use_ray=True, uniqueid="", hidden_channels=[64, 64],
             base_lr=0.1, decay_lr=0.5, cls="sage", aggr="mean",
             clamp_boundary=3, use_physical_loss=1, weighting="relative", seed=1,
             initial_epoch=0, hetero=True, model_file=None, num_workers_train=0):
    device = init_device(device, seed)
    if not uniqueid:
        uniqueid = uuid.uuid4()

    title = "analyze_constraints"
    max_epochs = 0
    val_batch_size = 1
    train_batch_size = 5
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    tolerance= 1e-4
    powerflow_tolerance = 0.01

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
                  "pin_memory": pin_memory,"tolerance":tolerance}

    all_param_hash = hashlib.md5(json.dumps(all_params).encode()).hexdigest()
    model_file = f"{save_path}/model_{uniqueid}_{seed}_{all_param_hash}.pt" if model_file is None else model_file
    figure_output_folder = f"{save_path}/figures_seed{seed}_{all_param_hash}/"
    os.makedirs(figure_output_folder, exist_ok=True)

    if experiment is not None:
        experiment.log_parameters(
            {**all_params, "model_file": model_file, "figure_output_folder": figure_output_folder})

    train_case_name, val_case_name, nb_graph, mutation_rate, mutations, train_graphs, train_networks, val_graphs, valid_networks = init_dataset(
        training_cases, validation_case, all_params, experiment, filter_dataset=False)

    if len(val_graphs) == 0:
        return None, train_graphs, train_networks, val_graphs, valid_networks

    graph_y = val_graphs[0]

    training_params = {"decay_lr": decay_lr, "base_lr": base_lr, "cls": cls, "hidden_channels": hidden_channels,
                       "aggr": aggr,
                       "train_case_name": train_case_name, "val_case_name": val_case_name}
    if experiment is not None:
        experiment.log_parameters(training_params)

    model = init_model(graph_y, all_params, aggr, hidden_channels, cls, model_file)

    model, (output, losses) = run_train_eval(model, train_graphs, train_networks, val_graphs, valid_networks,
                                             all_params, training_params, experiment, logging=False)
    grid_metrics = []
    output_nodes = {node: torch.cat([e[node] for e in output], 0) for node in y_nodes}
    nb_gens = {node: len(valid_networks.get("mutants")[0][node]) for node in y_nodes}
    neighboorhood = None
    for i, net in enumerate(valid_networks.get("mutants")):
        grid_metric = {"weighted": losses[i].get("weighted").mean(), "cost": losses[i].get("cost").mean()
            , "physical": losses[i].get("physical").mean(), "boundary": np.hstack(losses[i].get("boundary")).mean()}


        boundary_constraints = get_boundary_constraints_violations(i, net, y_nodes, output[i], nb_gens, opf, tolerance=tolerance)

        data = val_graphs[i][0]
        physical_loss, neighboorhood, duration, details = power_imbalance_loss(data, output[i], neighboorhood=neighboorhood,
                                                                         version="2")

        sn_mva = 100
        powerflow_constraints_P = abs(details.get("Pi") - details.get("Pi_true"))/sn_mva < powerflow_tolerance
        powerflow_constraints_Q = abs(details.get("Qi") - details.get("Qi_true"))/sn_mva < powerflow_tolerance
        powerflow_constraints = np.multiply(powerflow_constraints_P.cpu().numpy().astype(np.float32), powerflow_constraints_Q.cpu().numpy().astype(np.float32))

        validation_constraints = {"gen":boundary_constraints.get("gen").squeeze(0),"ext_grid":boundary_constraints.get("ext_grid")
                                  ,"bus":np.multiply(boundary_constraints.get("bus").astype(np.float32),powerflow_constraints).squeeze(0)
                                  ,"bus_boundary":boundary_constraints.get("bus").astype(np.float32).squeeze(0)
                                  ,"bus_powerflow":powerflow_constraints}

        validation_constraints_gen = dict(zip([f"gen_{i}" for i in range(len(validation_constraints.get("gen")))],validation_constraints.get("gen").tolist()))
        validation_constraints_ext_grid = dict(zip([f"ext_grid_{i}" for i in range(len(validation_constraints.get("ext_grid")))],validation_constraints.get("ext_grid").tolist()))
        validation_constraints_bus = dict(zip([f"bus_{i}" for i in range(len(validation_constraints.get("bus")))],validation_constraints.get("bus").tolist()))
        validation_boundaries_bus = dict(zip([f"bus_{i}_bd" for i in range(len(validation_constraints.get("bus_boundary")))],
                                              validation_constraints.get("bus_boundary").tolist()))
        validation_powerflow_bus = dict(zip([f"bus_{i}_pf" for i in range(len(validation_constraints.get("bus_powerflow")))],
                                              validation_constraints.get("bus_powerflow").tolist()))

        grid_metrics.append({**grid_metric, **validation_constraints_gen,**validation_constraints_ext_grid,**validation_constraints_bus,**validation_boundaries_bus,**validation_powerflow_bus})

    if experiment is not None:
        [experiment.log_metrics(e, prefix="graph_", step=i) for (i, e) in
         enumerate(grid_metrics)]

    return model, train_graphs, train_networks, val_graphs, valid_networks
