import os
try:
    from src.utils.appconfig import COMET_APIKEY
except ImportError:
    # load for env variable
    COMET_APIKEY = os.environ.get("COMET_APIKEY")
from comet_ml import Experiment
import time
import torch
from itertools import chain
import numpy as np
import pandas as pd
import json
import os

class NumpyEncoder(json.JSONEncoder):
    import numpy as np
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def init_comet(args, project_name="gnn", workspace="yamizi"):

    # You can define the workspace directly using env variables
    workspace = os.environ.get("COMET_WORKSPACE", workspace)
    timestamp = time.time()
    args["timestamp"] = timestamp
    experiment = Experiment(api_key=COMET_APIKEY,
                            project_name=project_name,
                            workspace=workspace,
                            auto_param_logging=False, auto_metric_logging=False,
                            parse_args=False, display_summary=False, disabled=False)
    experiment.log_parameters(args)

    if project_name != "debug":
        import warnings
        warnings.filterwarnings("ignore")

    return experiment


def log_dataframe(dic, experiment, name, base=0, limit=0):
    initial_val = len(list(dic.values())[0])
    equal_nb = all([initial_val == len(val) for val in dic.values()])
    if equal_nb:
        experiment.log_dataframe_profile(pd.DataFrame(dic), name=f"{name}_{base}")
    else:
        for (e, v) in dic.items():
            experiment.log_dataframe_profile(pd.DataFrame(v), name=f"{name}_{base}_{e}")


def log_dict_series(dic, experiment, limit=0):
    counter = 0
    time.sleep(60)
    for (e, v) in dic.items():
        for i, l in enumerate(v):
            if limit > 0 and i > limit:
                break
            experiment.log_metric(e, l, step=i)
            counter += 1

        if counter > 9000:
            time.sleep(60)
            counter = 0


def build_log_opf(dic, label, node, outputs, output_index, ground_truth, gt_index, delta=1e-10):
    dic = {**dic,
           f"{label}_pred_" + node: outputs[:, output_index].cpu().numpy(),
           f"{label}_true_" + node: ground_truth[:, gt_index].cpu().numpy(),
           }

    SE = (outputs[:, output_index].cpu().numpy() - ground_truth[:, gt_index].cpu().numpy()) ** 2
    relativeSE = SE / (ground_truth[:, gt_index].cpu().numpy() ** 2 + delta)

    AE = (outputs[:, output_index].cpu().numpy() - ground_truth[:, gt_index].cpu().numpy())
    relativeAE = AE / (ground_truth[:, gt_index].cpu().numpy() + delta)

    dic = {**dic, f"SE_{label}_" + node: SE, f"relativeSE_{label}_" + node: relativeSE, f"AE_{label}_" + node: AE,
           f"relativeAE_{label}_" + node: relativeAE}

    return dic


def log_opf(networks, val_graphs, outputs, y_nodes, experiment, hetero=True):
    (out_all, val_losses_all) = outputs
    # if hetero:
    output_nodes = {node: torch.cat([e[node] for e in out_all], 0) for node in y_nodes}
    labels = {node: torch.cat([e.data[node].y for e in val_graphs]) for node in y_nodes}
    # else:
    #     return {}
    #     ###TODO: rebuild all y nodes from bus vector
    #     bus_nodes = torch.cat([e for e in out_all], 0)
    #     y_nodes = ["ext_grid", "gen", "sgen"]
    #     nb_gens = {node: len(networks.get("original")[node]) for node in y_nodes}
    #     mask = list(chain.from_iterable([[e] * k for (e, k) in nb_gens.items()])) * len(networks.get("mutants"))
    #     output_nodes = {node: bus_nodes[np.array(mask) == node, :] for node in y_nodes}
    #
    #     mask_nan = ~torch.isnan(val_graphs[0].data.y)#.any(1)
    #     ground_truth = torch.cat([e.data.y[mask_nan] for e in val_graphs])
    #     labels = {node: ground_truth[np.array(mask) == node, :] for node in y_nodes}

    delta = 1e-10  # to avoid division by zero
    dic = {}
    for node, outputs in output_nodes.items():
        ground_truth = labels[node]
        if node == "gen":
            dic = build_log_opf(dic, "P", node, outputs, 0, ground_truth, 0)
            dic = build_log_opf(dic, "Q", node, outputs, 1, ground_truth, 1)

        elif node == "ext_grid":
            dic = build_log_opf(dic, "P", node, outputs, 2, ground_truth, 0)
            dic = build_log_opf(dic, "Q", node, outputs, 3, ground_truth, 1)

        elif node == "bus":
            dic = build_log_opf(dic, "Vm", node, outputs, 4, ground_truth, 0)
            dic = build_log_opf(dic, "Va", node, outputs, 5, ground_truth, 1)

        if node == "line":
            dic = build_log_opf(dic, "pl_mw", node, outputs, 6, ground_truth, 0)
            dic = build_log_opf(dic, "SE_ql", node, outputs, 7, ground_truth, 1)

            dic = build_log_opf(dic, "i_from_ka", node, outputs, 7, ground_truth, 2)
            dic = build_log_opf(dic, "i_to_ka", node, outputs, 8, ground_truth, 3)

    if experiment:
        log_dict_series(dic, experiment)

    return dic
