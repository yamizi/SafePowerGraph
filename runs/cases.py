import os.path
import sys
import hashlib

sys.path.append(".")
import uuid

from utils.logging import log_dict_series, log_opf, NumpyEncoder
from utils.pandapower import build_dataset, clear_duplicates
from utils.pandapower.opf_validation import validate_opf
from torch_geometric.nn import to_hetero

from torch_geometric.loader import DataLoader
from utils.models import GNN
from utils.custom_models import GPS
import torch
from utils.train import train_opf, train_cv
import json, os
import pickle

import random
import numpy as np
from huggingface_hub import hf_hub_download

from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T

from utils import hetero_to_homo

def add_random_walk_pe_to_hetero(
    hetero_data: Data,
    walk_length: int = 4,
    attr_name: str = 'pe',
    hetero:bool = True
):
    rw_pe_transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name=attr_name)
    if not hetero:
        return rw_pe_transform(hetero_data)
    for node_type in hetero_data.node_types:
        # Collect all edges where the target is this node type, possibly from multiple edge types
        combined_edges = []
        for edge_type in hetero_data.edge_types:
            if edge_type[2] == node_type:
                edges = hetero_data[edge_type].edge_index
                combined_edges.append(edges)
        if len(combined_edges) == 0:
            continue  # No incoming edges for this node type
        # Concatenate all incoming edges
        edge_index = torch.cat(combined_edges, dim=1)
        num_nodes = hetero_data[node_type].num_nodes
        sub_data = Data(x=torch.zeros((num_nodes, 1)), edge_index=edge_index)
        sub_data = rw_pe_transform(sub_data)
        hetero_data[node_type][attr_name] = sub_data[attr_name]
    return hetero_data



def init_dataset(training_cases, validation_case, all_params, experiment, filter_dataset=True):
    pickle_file = all_params.get("pickle_file")
    dataset_type = all_params.get("dataset_type")
    save_path = all_params.get("save_path")
    scale = all_params.get("scale")
    device = all_params.get("device")
    opf = all_params.get("opf")
    use_ray = all_params.get("use_ray")
    y_nodes = all_params.get("y_nodes")
    uniqueid = all_params.get("uniqueid")
    hetero = all_params.get("hetero")

    if (os.path.exists(pickle_file)):
        with open(pickle_file, "rb") as pickled:
            print("Loading existing dataset from", pickle_file)
            loaded = pickle.load(pickled)
        train_graphs, train_networks, val_graphs, valid_networks = loaded.get("train_graphs"), loaded.get(
            "train_networks"), loaded.get("val_graphs"), loaded.get("valid_networks")
        train_case_name, nb_graph, mutation_rate, mutations = training_cases[0]
        val_case_name, nb_graphs, mutation_rate, mutations = validation_case

    else:

        common_params = {"dataset_type": dataset_type, "save_dataframes": save_path, "experiment": experiment,
                         "scale": scale, "device": device, "opf": opf, "use_ray": use_ray, "y_nodes": y_nodes,
                         "hetero": hetero}

        val_case_name, nb_graphs, mutation_rate, mutations = validation_case
        val_graphs, valid_networks, _, _ = build_dataset(val_case_name, nbsamples=nb_graphs,
                                                         mutation_rate=mutation_rate,
                                                         uniqueid="{}/val".format(uniqueid),
                                                         mutations=mutations, **common_params)
        print("Correct validation graphs {}/{}".format(len(val_graphs), nb_graphs))

        train_graphs = []
        train_networks = {"original": [], "mutants": [], "convergence_time": []}
        nb_graphs = 0
        train_case_name = ""
        for training_case in training_cases:
            train_case_name, nb_graph, mutation_rate, mutations = training_case

            train_graph, train_network, _, _ = build_dataset(train_case_name, nbsamples=nb_graph,
                                                             mutation_rate=mutation_rate,
                                                             uniqueid="{}/train".format(uniqueid),
                                                             mutations=mutations, **common_params)
            train_graphs += train_graph
            train_networks["mutants"] += train_network["mutants"]
            train_networks["convergence_time"] += train_network["convergence_time"]
            train_networks["original"] += [train_network["original"]]
            nb_graphs += len(train_graph)

        if filter_dataset:
            train_graphs, train_networks, val_graphs, valid_networks = clear_duplicates(train_graphs, train_networks,
                                                                                        val_graphs, valid_networks)

        print("Correct training graphs {}/{}".format(len(train_graphs), nb_graphs))
        [experiment.log_metric("convergence_train_graphs", e, i) for (i, e) in
         enumerate(train_networks["convergence_time"])]

        with open(pickle_file, "ab") as f:
            pickle.dump({"train_graphs": [e.to("cpu") for e in train_graphs], "train_networks": train_networks,
                         "val_graphs": [e.to("cpu") for e in val_graphs],
                         "valid_networks": valid_networks}, f)

    experiment.log_metric("nb_train_graphs", len(train_graphs))
    experiment.log_metric("nb_valid_graphs", len(val_graphs))


    return train_case_name, val_case_name, nb_graph, mutation_rate, mutations,  train_graphs, train_networks, val_graphs, valid_networks


def run_hp(train_graphs, all_params, experiment ):
    pickle_file = all_params.get("pickle_file")
    train_batch_size = all_params.get("train_batch_size")
    num_samples = all_params.get("num_samples")
    max_epochs = all_params.get("max_epochs")
    val_batch_size = all_params.get("val_batch_size")
    clamp_boundary = all_params.get("clamp_boundary")
    use_physical_loss = all_params.get("use_physical_loss")
    device = all_params.get("device")
    base_lr = all_params.get("base_lr")
    cv_ratio = all_params.get("cv_ratio")
    y_nodes = all_params.get("y_nodes")
    graph_y = train_graphs[0]

    num_graphs = min(1000, len(train_graphs))
    print("Hyper parameter tuning model with device", device)
    best_config, metrics_dataframe = train_cv(pickle_file, cv_ratio, y_nodes=y_nodes, device=device,
                                              base_lr=[base_lr / 100, base_lr],
                                              max_epochs=max_epochs // 5, graph=graph_y, num_samples=num_samples,
                                              num_graphs=num_graphs,train_batch_size=train_batch_size,
                                              val_batch_size=val_batch_size, train_graphs=train_graphs,
                                              clamp_boundary=clamp_boundary, use_physical_loss=use_physical_loss
                                              )
    best_config["num_graphs"] = num_graphs
    if experiment is not None:
        [experiment.log_dataframe_profile(df, v) for (df, v) in metrics_dataframe.items()]

    experiment.log_parameters(best_config, prefix="hp_")
    decay_lr = best_config["decay_lr"]
    base_lr = best_config["lr"]
    hidden_channels = [best_config["hidden_channels"]] * best_config["nb_hidden_layers"]
    cls = best_config["cls"]
    aggr = best_config["aggr"]

    return decay_lr, base_lr, hidden_channels, cls, aggr

def init_model(graph_y, all_params, aggr, hidden_channels, cls, model_file):
    device = all_params.get("device")
    hetero = all_params.get("hetero")

    data = graph_y[0]

    if cls=="gps":
        model = GPS(hidden_channels=hidden_channels, out_channels=graph_y.num_outputs)
    else:
        model = GNN(hidden_channels=hidden_channels, out_channels=graph_y.num_outputs, aggr=aggr, cls=cls)

    if hetero:
        model = to_hetero(model, data.metadata(), aggr='sum').to(device)

    if (os.path.exists(model_file)):
        print("Loading existing pre-trained model from", model_file)
        weights = torch.load(model_file, map_location="cpu")
        with torch.no_grad():
            model(data.x_dict, data.edge_index_dict)
        model.load_state_dict(weights)

    return model

def init_device(device, seed):
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        device = "cpu"

    if torch.cuda.is_available() and "cuda" in device:
        device = device
        print("running with ", device)
    else:
        device = "cpu"
        print("running with cpu backend")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    return device

def run_train_eval(model, train_graphs, train_networks, val_graphs, valid_networks, all_params, training_params, experiment, logging=True):
    pickle_file = all_params.get("pickle_file")
    train_batch_size = all_params.get("train_batch_size")
    pin_memory = all_params.get("pin_memory")
    max_epochs = all_params.get("max_epochs")
    val_batch_size = all_params.get("val_batch_size")
    clamp_boundary = all_params.get("clamp_boundary")
    use_physical_loss = all_params.get("use_physical_loss")
    device = all_params.get("device")
    hetero = all_params.get("hetero", )
    num_workers_train = all_params.get("num_workers_train")
    y_nodes = all_params.get("y_nodes")
    weighting = all_params.get("weighting")
    use_ray = all_params.get("use_ray")
    save_path = all_params.get("save_path")
    uniqueid = all_params.get("uniqueid")
    opf = all_params.get("opf")
    walk_length = all_params.get("walk_length",4)
    decay_lr = training_params.get("decay_lr")
    base_lr = training_params.get("base_lr")
    cls = training_params.get("cls")

    train_list = [g.set_device(device, num_workers_train)[0] for g in train_graphs]
    validation_list = [g.set_device(device, num_workers_train)[0] for g in val_graphs]
    node_types = validation_list[0].node_types
    if not hetero:
        train_list = hetero_to_homo(train_list)
        validation_list = hetero_to_homo(validation_list)

    if cls=="gps":
        train_list = [add_random_walk_pe_to_hetero(g, walk_length=walk_length, hetero=hetero) for g in train_list]
        validation_list = [add_random_walk_pe_to_hetero(g, walk_length=walk_length, hetero=hetero) for g in validation_list]

    train_loader = DataLoader(train_list, batch_size=train_batch_size, num_workers=num_workers_train,
                              pin_memory=pin_memory)
    val_loader = DataLoader(validation_list, batch_size=val_batch_size, pin_memory=pin_memory)

    train_losses, val_losses, val_losses_nodes, last_out, b_train_losses, p_train_losses, b_val_losses, lr = train_opf(
        model, train_loader, val_loader, max_epochs=max_epochs, y_nodes=y_nodes, device=device,
        base_lr=base_lr, decay_lr=decay_lr, experiment=experiment, clamp_boundary=clamp_boundary,
        use_physical_loss=use_physical_loss, weighting=weighting, hetero=hetero, node_types=node_types)

    val_losses_gen, val_losses_ext_grid, val_losses_bus, val_losses_line = val_losses_nodes
    constrained_networks, errors_network = validate_opf(valid_networks, val_graphs, last_out, y_nodes=y_nodes, opf=opf,
                                                        use_ray=use_ray, hetero=hetero)

    if logging:
        print("Logging results")

        log_dict = {"constraint_boundary": constrained_networks[:, 1].tolist(),
                    "constraint_opf": constrained_networks[:, 0].tolist(),
                    "constraint": constrained_networks.prod(1).tolist()}
        epoch_dict = {"train_losses": train_losses, "val_losses": val_losses,
                      "b_train_losses": b_train_losses,
                      "b_train_losses": b_train_losses, "b_val_losses": b_val_losses, "learning_rate": lr,
                      "val_losses_gen": val_losses_gen, "val_losses_ext_grid": val_losses_ext_grid,
                      "val_losses_bus": val_losses_bus, "val_losses_line": val_losses_line}

        losses_file = f"{save_path}/{uniqueid}_losses.json"
        with open(losses_file, "w") as outfile:
            json.dump(epoch_dict, outfile)

        constraints_file = f"{save_path}/{uniqueid}_constraints.json"
        with open(constraints_file, "w") as outfile:
            json.dump(log_dict, outfile)

        relativeSE = log_opf(valid_networks, val_graphs, last_out, y_nodes, None, hetero=hetero)
        errors_file = f"{save_path}/{uniqueid}_errors.json"
        with open(errors_file, "w") as outfile:
            json.dump(relativeSE, outfile, cls=NumpyEncoder)

        if experiment is not None:
            experiment.log_asset(pickle_file) # dataset used to train / evaluate
            experiment.log_asset(losses_file)
            experiment.log_asset(constraints_file)
            experiment.log_asset(errors_file)

            log_dict_series(log_dict, experiment, 1000)
            log_dict_series(relativeSE, experiment, 1000)


    return model, last_out
def run_case(training_cases=[["case9", 64, 0.7, ["cost", "load"]]], experiment=None,
             validation_case=["case9", 64, 0.7, ["cost", "load"]], plot=True,
             save_path="../output", title="", dataset_type="y_OPF",
             scale=False,
             max_epochs=500, y_nodes=["gen", "ext_grid", "bus"], train_batch_size=5, val_batch_size=5,
             device="cuda", filter_dataset=True, opf=2, use_ray=True, uniqueid="", hidden_channels=[64, 64],
             base_lr=0.1, decay_lr=0.5, cv_ratio=0, cls="sage", aggr="mean", num_samples=100, build_db_only=False,
             clamp_boundary=0, use_physical_loss=1, weighting="relative", seed=20, return_model_if_exists=False,
             initial_epoch=0, hetero=True, model_file=None, model_output_file=None, num_workers_train=0,
             token="", args=None):

    device = init_device(device, seed)
    if not uniqueid:
        uniqueid = uuid.uuid4()

    os.makedirs(save_path, exist_ok=True)
    case = training_cases[0][0]              # e.g., "case9"
    mutation_type = training_cases[0][3][0]  # e.g., "cost"
    filename = f"OPF_{case}_{mutation_type}_{seed}.pkl"
    pickle_file = f"{save_path}/{uniqueid}_{seed}_{hetero}.pkl"
    if not os.path.exists(pickle_file):
        try:
            token = os.environ.get("HUGGINFACE_TOKEN",token)
            downloaded_path = hf_hub_download(
                repo_id="LISTTT/NeurIPS_2025_BMDB",
                filename=filename,
                repo_type="dataset",
                cache_dir=save_path,  
                token=token   
            )
            
            os.rename(downloaded_path, pickle_file)
            print(f"Downloaded from HuggingFace and saved to: {pickle_file}")
        
        except Exception as e:
            print(f"Download failed from Hugging Face: {e}")
    else:
        print(f"Found existing pickle file: {pickle_file}")
    
    pin_memory = device == "cpu" or num_workers_train != 0
    num_workers_train = int(num_workers_train)

    # Running the experiment only for the remaining epochs
    max_epochs = max_epochs - initial_epoch

    all_params = {"uniqueid": uniqueid, "max_epochs": max_epochs, "dataset_type": dataset_type,
                  "scale": scale, "hetero": hetero, "opf": opf, "use_ray": use_ray,
                  "save_path": save_path, "title": title, "y_nodes": y_nodes, "plot": plot,
                  "device": device, "train_batch_size": train_batch_size,
                  "hidden_channels": hidden_channels, "num_samples": num_samples,
                  "val_batch_size": val_batch_size, "pickle_file": pickle_file,
                  "clamp_boundary": clamp_boundary, "use_physical_loss": use_physical_loss,
                  "base_lr": base_lr, "cv_ratio": cv_ratio, "cls": cls, "aggr": aggr,
                  "weighting": weighting, "losses": "mse+l1", "seed": seed,
                  "initial_epoch": initial_epoch, "num_workers_train": num_workers_train,
                  "pin_memory": pin_memory, "hetero":args.hetero}

    all_param_hash = hashlib.md5(json.dumps(all_params).encode()).hexdigest()
    model_file = f"{save_path}/model_{uniqueid}_{seed}_{all_param_hash}.pt" if (model_file is None or model_file=="") else model_file
    model_output_file = f"{save_path}/model_{uniqueid}_{seed}_{all_param_hash}.pt" if model_output_file is None else model_output_file

    if experiment is not None:
        experiment.log_parameters({**all_params, "model_file": model_file, "model_output_file": model_output_file})

    train_case_name, val_case_name, nb_graph, mutation_rate, mutations, train_graphs, train_networks, val_graphs, valid_networks = init_dataset(training_cases, validation_case, all_params, experiment, filter_dataset=filter_dataset)

    if build_db_only:
        return None, train_graphs, train_networks, val_graphs, valid_networks


    if len(train_graphs) == 0:
        return None, train_graphs, train_networks, val_graphs, valid_networks

    graph_y = train_graphs[0]

    if cv_ratio > 0:
        decay_lr, base_lr, hidden_channels, cls, aggr = run_hp(train_graphs, all_params, experiment )

    training_params = {"decay_lr":decay_lr,"base_lr":base_lr,"cls": cls, "hidden_channels": hidden_channels, "aggr": aggr,
                       "train_case_name":train_case_name,"val_case_name":val_case_name}
    if experiment is not None:
        experiment.log_parameters(training_params)

    model = init_model(graph_y, all_params, aggr, hidden_channels, cls, model_file)

    if return_model_if_exists:
        return model, train_graphs, train_networks, val_graphs, valid_networks

    model, (output, losses)  = run_train_eval(model, train_graphs, train_networks, val_graphs, valid_networks, all_params, training_params, experiment)
    torch.save(model.state_dict(), model_output_file)
    print("Model saved to", model_output_file)

    return model, train_graphs, train_networks, val_graphs, valid_networks
