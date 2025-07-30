import numpy as np
import torch

MIN_PQ = -200
MAX_PQ = 1800

MAX_ANGLE = 1

def normalizeCols(dataset, case="", columns=["p_mw", "q_mvar"], min_val=MIN_PQ, max_val=MAX_PQ):
    dts = []
    columns_to_affect = [e for e in columns if e in dataset.columns]
    if len(columns_to_affect) == 0:
        return dataset

    # max_vals = dataset[columns_to_affect].max()
    # min_vals = dataset[columns_to_affect].min()

    # if case == "case13" or case == "case123":
    #     min_vals = min_val
    #     max_vals = 5000.0 / 3  # Base kVA
    # elif case == "case8500":
    #     min_vals = min_val
    #     max_vals = 27.5 * 1000 / 3  # Base kVA
    # else:
    #     min_vals = min_val
    #     max_vals = max_val

    min_vals = min_val
    max_vals = max_val

    dataset[columns_to_affect] = (dataset[columns_to_affect] - min_vals) / (max_vals - min_vals)
    return dataset


def denormalize_outputs(out, sn_mva, angle="deg",type="hgnn"):

    if type=="hgnn":
        out["gen"] = out["gen"]*sn_mva
        out["sgen"] = out["sgen"] * sn_mva
        out["ext_grid"] = out["ext_grid"] * sn_mva
        out["bus"][:,1] = torch.rad2deg(out["bus"][:,1]) if angle=="deg" else out["bus"][:,1]

    else:
        print("we need to split the components")
    return out
