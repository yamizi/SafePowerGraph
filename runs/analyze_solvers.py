import os.path
import sys
import hashlib

sys.path.append(".")
import uuid
import pandas as pd
import numpy as np
from copy import deepcopy
import json, os
from runs.cases import init_dataset, init_device
import pandapower as pp
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
from runs.matpower import opf as matpower_opf
from runs.opendss import pf as opendss_pf

def run_case(experiment=None,
             training_cases=[["case9", 64, 0.7, ["cost", "load"]]],
             validation_case=["case9", 64, 0.7, ["cost", "load"]], plot=True,
             save_path="../output", dataset_type="y_OPF",
             scale=False, y_nodes=["gen", "ext_grid", "bus"],
             device="cuda", opf="pandapower_opf", use_ray=True, uniqueid="", seed=1,
             initial_epoch=0, hetero=True, model_file="",num_workers_train=0, ref_pf="matpower_opf"):
    device = init_device(device, seed)
    if not uniqueid:
        uniqueid = uuid.uuid4()

    title = "analyze_solvers"
    max_epochs = 0
    val_batch_size = 1
    train_batch_size = 5
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    pickle_file = f"{save_path}/{uniqueid}_{seed}_{hetero}.pkl"
    pin_memory = device == "cpu" or num_workers_train != 0
    num_workers_train = int(num_workers_train)

    all_params = {"uniqueid": uniqueid, "max_epochs": max_epochs, "dataset_type": dataset_type,
                  "scale": scale, "hetero": hetero, "opf": ref_pf, "use_ray": use_ray,
                  "save_path": save_path, "title": title, "y_nodes": y_nodes, "plot": plot,
                  "device": device, "train_batch_size": train_batch_size,
                  "val_batch_size": val_batch_size, "pickle_file": pickle_file,"losses": "mse+l1", "seed": seed,
                  "initial_epoch": initial_epoch, "num_workers_train": num_workers_train,
                  "pin_memory": pin_memory}

    all_param_hash = hashlib.md5(json.dumps(all_params).encode()).hexdigest()
    figure_output_folder = f"{save_path}/figures_seed{seed}_{all_param_hash}/"
    os.makedirs(figure_output_folder, exist_ok=True)


    train_case_name, val_case_name, nb_graph, mutation_rate, mutations, train_graphs, train_networks, val_graphs, valid_networks = init_dataset(
        training_cases, validation_case, all_params, experiment, filter_dataset=False)

    all_gen = []
    all_ext_grid = []
    all_bus = []

    diff = []

    ref_bus = ref_gen = ref_ext_grid = None

    for i, original_network in enumerate(valid_networks.get("mutants")):
        res_gen = original_network.res_gen[["p_mw","q_mvar"]]
        ref_gen = deepcopy(res_gen)
        res_gen["method"] = "matpower_opf"

        res_ext_grid = original_network.res_ext_grid[["p_mw","q_mvar"]]
        ref_ext_grid = deepcopy(res_ext_grid)
        res_ext_grid["method"] = "matpower_opf"

        res_bus = original_network.res_bus[["vm_pu","va_degree"]]
        res_bus["va_degree"] = np.deg2rad(res_bus["va_degree"])
        ref_bus = deepcopy(res_bus)
        res_bus["method"] = "matpower_opf"

        res_gen["graph"] = res_ext_grid["graph"] = res_bus["graph"] = i


        all_gen = pd.concat([all_gen, res_gen]) if i>0 else res_gen
        all_ext_grid = pd.concat([all_ext_grid, res_ext_grid])  if i>0 else res_ext_grid
        all_bus = pd.concat([all_bus, res_bus])  if i>0 else res_bus

        for o in opf.split("+"):
            network = deepcopy(original_network)
            #try:
            if o=="powermodel_opf":
                pp.runpm_ac_opf(network)

            elif o=="pandapower_opf":
                pp.runopp(network)

            elif o=="opendss_pf":
                loads = network.mutated_loads
                lines_in_service = network.lines_in_service
                network, convergence_time = opendss_pf(case=val_case_name,
                                                         all_loads=[loads] if loads is not None else None,
                                                         lines_in_service=lines_in_service, network=network)
                network = network[0]
                if network is None:
                    continue

            elif o=="pandapower_pf":
                pp.runpp(network)
            # except Exception as e:
            #     print(i, ": error pf", e)
            #     continue

            res_gen = network.res_gen[["p_mw","q_mvar"]]
            d_gen = abs((res_gen/100)**2 - (ref_gen/100)**2)

            res_ext_grid = network.res_ext_grid[["p_mw","q_mvar"]]
            d_ext_grid = abs((res_ext_grid / 100) ** 2 - (ref_ext_grid / 100) ** 2)

            res_bus = network.res_bus[["vm_pu", "va_degree"]]
            res_bus["va_degree"] = np.deg2rad(res_bus["va_degree"])
            d_bus = abs(res_bus - ref_bus)


            res_gen["method"] = o
            res_ext_grid["method"] = o
            res_bus["method"] = o

            res_gen["graph"] = res_ext_grid["graph"] = res_bus["graph"] = i

            all_gen = pd.concat([all_gen,res_gen ])
            #all_gen["element"] = all_gen.index
            all_ext_grid = pd.concat([all_ext_grid,res_ext_grid])
            #all_ext_grid["element"] = all_ext_grid.index
            all_bus =pd.concat([all_bus,res_bus])
            #all_bus["element"] = all_bus.index

            diff.append({"graph": i, "$P_{gen}$": d_gen["p_mw"].mean(), "$Q_{gen}$": d_gen["q_mvar"].mean()
                        , "$P_{slack}$": d_ext_grid["p_mw"].mean(), "$Q_{slack}$": d_ext_grid["q_mvar"].mean()
                        , "$V$": d_bus["vm_pu"].mean(), r"$\theta$": d_bus["va_degree"].mean(),"method":o})

    grouped = pd.DataFrame(diff).drop(columns="graph").groupby("method").agg(["mean","std"])
    grouped = grouped.rename(index={"powermodel_opf":"PowerModels","pandapower_opf":"PandaPower",
                                    "opendss_pf":"OpenDSS","pandapower_pf":"PandaPower"})


    #print(grouped)
    """Plotting"""
    ax = grouped.plot.bar(logy=True, rot=0, title="")
    plt.xlabel("", rotation=0)
    ticks = [10**i for i in  range(np.log10(grouped.values.min()/10).round().astype(int), np.log10(grouped.values.max()*10).round().astype(int))]
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:])
    # Move the legend below the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

    # Adjust the layout
    plt.subplots_adjust(bottom=0.2)


    mutation = mutations[0].replace(":","_")
    extension = "png"
    figure_path =os.path.join(figure_output_folder,f"diff_{val_case_name}_{mutation}.{extension}")
    plt.savefig(figure_path, bbox_inches='tight')
    plt.savefig(os.path.join(figure_output_folder,f"diff_{val_case_name}_{mutation}.pdf"), bbox_inches='tight')

    if experiment is not None:
        experiment.log_parameters({**all_params,**all_ext_grid.groupby(["method"]).count()["graph"].to_dict(),"figure_path":figure_path})
        experiment.log_dataframe_profile(all_bus.reset_index(), "all_bus")
        experiment.log_dataframe_profile(all_ext_grid.reset_index(), "all_ext_grid")
        experiment.log_dataframe_profile(all_gen.reset_index(), "all_gen")
        experiment.log_dataframe_profile(grouped, "diff")
        experiment.log_image(figure_path, name=f"diff_{val_case_name}_{mutation}.{extension}", overwrite=False, step=i)


    return val_graphs, valid_networks
