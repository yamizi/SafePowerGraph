from typing import Callable, Optional
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset
)
import pandapower as pp

import numpy as np

from copy import deepcopy
import psutil, gc
import torch_geometric.transforms as T
import json
import uuid
import os
from utils.pandapower.mutations import mutate_costs, mutate_loads, disable_lines
import itertools
import ray
import time
from utils.pandapower.pandapower_graph import PandaPowerGraph
from runs.matpower import opf as matpower_opf
from runs.opendss import pf as opendss_pf
import copy


def clear_duplicates(train_graphs, train_networks, val_graphs, valid_networks):
    print("clearing duplicates")
    train_graphs_c = deepcopy(train_graphs)

    val_str = [val_graph.data.to_dict().__str__() for val_graph in val_graphs]
    train_str = [train_graph.data.to_dict().__str__() for train_graph in train_graphs_c]

    comparisons = np.array([a == b for (a, b) in itertools.product(val_str, train_str)]).reshape(len(val_str),
                                                                                                 len(train_str))

    nb_duplicates = np.sum(comparisons)
    print("Found ", nb_duplicates, " duplicates")

    correct = np.where(comparisons.sum(0) == 0)[0]
    train_graphs_filtered = [train_graphs[i] for i in correct]
    train_mutants_filtered = [train_networks.get("mutants")[i] for i in correct]
    train_convergence_filtered = [train_networks.get("convergence_time")[i] for i in correct]

    train_networks_filtered = {"mutants": train_mutants_filtered, "original": train_networks.get("original"),
                               "convergence_time": train_convergence_filtered}
    return train_graphs_filtered, train_networks_filtered, val_graphs, valid_networks


def build_batch_graph(sample_id, network, mutations, mutation_rate, opf, transforms, scale, dataset_type, hetero,
                      save_dataframes, case, uniqueid, experiment, batch_size=25, y_nodes=[], device="cpu"):

    graphs = []

    loads = [mutate_loads(copy.deepcopy(network), mutation_rate=mutation_rate, relative=True)[1] for i in
             range(batch_size)] if "load_relative" in mutations else None
    lines_in_service = [disable_lines(copy.deepcopy(network), mutation_rate=mutation_rate, num_lines_disable=1)[1] for i
                        in range(batch_size)] if "line_nminus1" in mutations else None

    octave_path = os.environ.get("OCTAVE_PATH", None)
    # try:
    if opf=="opendss_pf":
        networks, convergence_times = opendss_pf(case=case, all_loads=loads, lines_in_service=lines_in_service,
                                                   octave_path=octave_path,
                                                   batch_size=batch_size)
    elif opf=="matpower_pf":
        networks, convergence_times = matpower_opf(case=case, all_loads=loads, lines_in_service=lines_in_service,
                                                   octave_path=octave_path,
                                                   batch_size=batch_size)
    else:
        networks, convergence_times = matpower_opf(case=case, all_loads=loads, lines_in_service=lines_in_service,
                                                   octave_path=octave_path,
                                                   batch_size=batch_size)
    for i, network in enumerate(networks):
        if network is None:
            continue


        network.mutated_loads = loads[i] if loads and len(loads) else None
        network.lines_in_service = lines_in_service[i] if lines_in_service and len(lines_in_service) else None

        graph_y = PandaPowerGraph(network, include_res=True, opf_as_y=True, preprocess='metapath2vec',
                                  transform=T.Compose(transforms), scale=scale, hetero=hetero,
                                  y_nodes=y_nodes, device=device)

        graphs.append((graph_y, network, convergence_times[i]))

    return graphs


@ray.remote
def build_one_graph_ray(sample_id, original_network, mutations, mutation_rate, opf, transforms, scale, dataset_type,
                        hetero,
                        save_dataframes, case, uniqueid, experiment, y_nodes, device="cpu"):
    return build_one_graph(sample_id, original_network, mutations, mutation_rate, opf, transforms, scale, dataset_type,
                           hetero,
                           save_dataframes, case, uniqueid, experiment, y_nodes, device)


def build_one_graph(sample_id, network, mutations, mutation_rate, opf, transforms, scale, dataset_type, hetero,
                    save_dataframes, case, uniqueid, experiment, y_nodes, device="cpu"):
    gc.collect()
    loads = None
    lines_in_service = None
    convergence_time = 0
    mutations_type = [m.split(":")[0] for m in mutations ]
    network.mutated_loads = None
    network.lines_in_service = None
    if mutation_rate > 0:
        if "cost" in mutations_type:
            _index = mutations_type.index("cost")
            params = mutations[_index].split(":")
            kwargs = {}
            rate = mutation_rate
            if len(params)>1:
                kwargs["min_cost"] = float(params[1])
            if len(params)>2:
                kwargs["max_cost"] = float(params[2])
            if len(params)>3:
                rate = float(params[3])

            network, masked = mutate_costs(copy.deepcopy(network), mutation_rate=rate, **kwargs)

        if "load_relative" in mutations_type:
            _index = mutations_type.index("load_relative")
            params = mutations[_index].split(":")
            kwargs = {}
            rate = mutation_rate
            if len(params) > 1:
                kwargs["min_p"] = float(params[1])
            if len(params) > 2:
                kwargs["max_p"] = float(params[2])
            if len(params) > 3:
                kwargs["min_q"] = float(params[3])
            if len(params) > 4:
                kwargs["max_q"] = float(params[4])
            if len(params) > 5:
                rate = float(params[5])

            network, loads = mutate_loads(copy.deepcopy(network), mutation_rate=rate, relative=True, **kwargs)
            network.mutated_loads = loads

        elif "load" in mutations_type:

            _index = mutations_type.index("load")
            params = mutations[_index].split(":")
            kwargs = {}
            rate = mutation_rate
            if len(params) > 1:
                kwargs["min_p"] = float(params[1])
            if len(params) > 2:
                kwargs["max_p"] = float(params[2])
            if len(params) > 3:
                kwargs["min_q"] = float(params[3])
            if len(params) > 4:
                kwargs["max_q"] = float(params[4])
            if len(params) > 5:
                rate = float(params[5])

            network, loads = mutate_loads(copy.deepcopy(network), mutation_rate=rate, **kwargs)
            network.mutated_loads = loads


        if "line_nminus1" in mutations_type:
            _index = mutations_type.index("line_nminus1")
            params = mutations[_index].split(":")
            kwargs = {}
            rate = mutation_rate
            if len(params) > 1:
                kwargs["num_lines_disable"] = int(params[1])
            if len(params) > 2:
                rate = float(params[2])
            network, lines_in_service = disable_lines(copy.deepcopy(network), mutation_rate=rate, **kwargs)
            network.lines_in_service = lines_in_service

    if str(opf)=="matpower_opf" or ("opf" not in str(opf) and int(opf) == 3):
        octave_path = os.environ.get("OCTAVE_PATH", None)
        try:
            network, convergence_time = matpower_opf(case=case, all_loads=[loads] if loads is not None else None, lines_in_service=lines_in_service,
                                                     octave_path=octave_path, network=network)
            network = network[0]
            convergence_time = convergence_time[0]
            if network is None:
                return None, None, None
        except Exception as e:
            print("matpower opf error", e)
            return None, None, None

    elif str(opf)=="matpower_pf":
        octave_path = os.environ.get("OCTAVE_PATH", None)
        try:
            network, convergence_time = matpower_opf(case=case, all_loads=[loads] if loads is not None else None, lines_in_service=lines_in_service,
                                                     octave_path=octave_path, network=network, command="runpf")
            network = network[0]
            convergence_time = convergence_time[0]
            if network is None:
                return None, None, None
        except Exception as e:
            print("matpower pf error", e)
            return None, None, None

    elif str(opf)=="opendss_pf":

        try:

            l = [loads] if loads is not None else None
            network, convergence_time = opendss_pf(case=case, all_loads=l, lines_in_service=lines_in_service,
                                                   network=network)
            network = network[0]
            convergence_time = convergence_time[0]
            if network is None:
                return None, None, None
        except Exception as e:
            print("matpower pf error", e)
            return None, None, None

    else:
        # fix minimum r_ohm and clean diagnostic warning
        network.line.r_ohm_per_km = network.line.r_ohm_per_km.clip(0.011)

        try:
            run_errors = pp.diagnostic(copy.deepcopy(network), report_style="compact")
            network.original_errors = run_errors
            print(run_errors)
            init = time.time()
            if ("opf" not in str(opf) and int(opf) == 2)  or opf=="powermodel_opf":
                pp.runpm_ac_opf(network)
            elif ("opf" not in str(opf) and int(opf) == 1) or opf=="pandapower_opf":
                pp.runopp(network)

            elif ("opf" not in str(opf) and int(opf) == 0) or opf=="pandapower_pf":
                pp.runpp(network)

            else:
                raise ValueError("Unrecognized simulator")

            convergence_time = time.time() - init

            if not network.OPF_converged:
                print("not converged")
                return None, None, None

            if network.res_bus.isnull().values.any():
                print("Some buses did not run")
                return None, None, None


        except Exception as e:
            print("error in opf", e)
            return None, None, None

    if dataset_type == "y_no_OPF":
        graph_y = PandaPowerGraph(network, include_res=False, opf_as_y=True, preprocess='metapath2vec',
                                  transform=T.Compose(transforms), scale=scale, hetero=hetero, device=device)
    elif dataset_type == "y_OPF":
        graph_y = PandaPowerGraph(network, include_res=True, opf_as_y=True, preprocess='metapath2vec',
                                  transform=T.Compose(transforms), scale=scale, hetero=hetero, device=device)
    elif dataset_type == "no_y_OPF":
        graph_y = PandaPowerGraph(network, include_res=True, opf_as_y=False, preprocess='metapath2vec',
                                  transform=T.Compose(transforms), scale=scale, hetero=hetero, device=device)

    if False and save_dataframes is not None:
        path = "{}/{}_{}/".format(save_dataframes, case, uniqueid)
        graph_y.export(path + "op_{}".format(sample_id), experiment=experiment)

    return graph_y, network, convergence_time


def build_dataset(case="case9", nbsamples=20, dataset_type="y_OPF", save_dataframes=None, opf=1,
                  mutations=["cost", "load"], mutation_rate=0.7, uniqueid=None, experiment=None, scale=True,
                  hetero=True, device="cpu", use_ray=True, batch_size=100, y_nodes=["gen", "ext_grid", "bus", "line"]):
    print(f"building dataset with {nbsamples} variants, ray {use_ray} and device {device}")

    case_method = getattr(pp.networks, case)
    original_network = case_method()
    original_network.case = case
    networks = {"original": original_network, "mutants": [], "convergence_time": []}
    network = deepcopy(original_network)
    uniqueid = uuid.uuid4() if uniqueid is None else uniqueid
    path = "."

    transforms = [T.ToUndirected(merge=True)] if hetero else []
    transforms = [T.ToDevice(device)] + transforms
    if scale:
        transforms.append((T.NormalizeFeatures()))
        scale = False

    graphs = []
    sample_id = 0

    while len(graphs) < nbsamples and sample_id < nbsamples * 100:
        # stop if we mutated more than 100 times the size needed without finding enough valid examples
        print("loop valid sample id", sample_id, " total graphs", len(graphs))
        sample_id = sample_id + nbsamples

        if use_ray:
            graph_y_network = [
                build_one_graph_ray.remote(sample_id, original_network, mutations, mutation_rate, opf, transforms,
                                           scale, dataset_type, hetero,
                                           save_dataframes, case, uniqueid, y_nodes=y_nodes,
                                           experiment=None) for sample_id in
                range(nbsamples)]

            graph_y_network = ray.get(graph_y_network)
        else:

            if (opf== "matpower_opf" or opf== "matpower_pf" or  opf== "opendss_pf" or int(opf) == 3 ) and batch_size > 1:
                graph_y_network = []
                for i in range(len(graphs), nbsamples, batch_size):
                    print("batch generation ", i, "+", batch_size, "/", nbsamples)
                    step_network = build_batch_graph(sample_id, original_network, mutations, mutation_rate, opf,
                                                     transforms, scale, dataset_type, hetero,
                                                     save_dataframes, case, uniqueid, experiment=experiment,
                                                     batch_size=batch_size, y_nodes=y_nodes)
                    graph_y_network = graph_y_network + step_network
                    gc.collect()
            else:
                graph_y_network = [
                    build_one_graph(sample_id, original_network, mutations, mutation_rate, opf, transforms, scale,
                                    dataset_type, hetero, save_dataframes, case, uniqueid,
                                    y_nodes=y_nodes, experiment=experiment) for sample_id in
                    range(nbsamples)]

        graph_y_networks = [g for g in graph_y_network if g[0] is not None]
        graph_y, networks_y, convergence_times = list(zip(*graph_y_networks)) if len(graph_y_networks) else ([], [], [])

        graphs = graphs + list(graph_y)
        networks["mutants"] = networks["mutants"] + list(networks_y)
        networks["convergence_time"] = networks["convergence_time"] + list(convergence_times)

    networks["mutants"] = networks["mutants"][:nbsamples]
    networks["convergence_time"] = networks["convergence_time"][:nbsamples]
    graphs = graphs[:nbsamples]
    return graphs, networks, path, uniqueid
