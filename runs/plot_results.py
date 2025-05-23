import pandas as pd
import numpy as np
import json

columns = {"P_ext_grid": ("P_true_ext_grid", "P_pred_ext_grid"),
           "P_gen": ("P_true_gen", "P_pred_gen"),
           "Q_ext_grid": ("Q_true_ext_grid", "Q_pred_ext_grid"),
           "Q_gen": ("Q_true_gen", "Q_pred_gen"), }

root_folder = "../submissions/8000/hgnn"


# root_folder="../submissions/8000/gnn"
# root_folder="../submissions/8000/fcnn"
# root_folder="../submissions/800"

def run(name, val):
    column = columns.get(val)
    f = pd.read_json(name)
    f[["xp", "node"]] = f["name"].str.split(" ", expand=True)
    print(name, f.node.unique())
    df = f.explode(["x", "y"])
    pivot_df = df.pivot_table("y", index="x", columns="node")
    pivot_df["mse"] = ((pivot_df[column[0]] - pivot_df[column[1]]) ** 2)
    pivot_df["relative_mse"] = pivot_df.mse / pivot_df[column[1]]
    print("mse", pivot_df["mse"].mean(), pivot_df["mse"].std(), pivot_df["relative_mse"].mean(),
          pivot_df["relative_mse"].std())

    pivot_df["mae"] = ((pivot_df[column[0]] - pivot_df[column[1]])).abs()
    pivot_df["relative_mae"] = pivot_df.mae / pivot_df[column[1]]
    # print("mae", pivot_df["mae"].mean(), pivot_df["mae"].std(),pivot_df["relative_mae"].mean(), pivot_df["relative_mae"].std())
    return pivot_df.drop(columns=["mse", "mae", "relative_mse", "relative_mae"])


cases = ["P_ext_grid", "P_gen", "Q_ext_grid", "Q_gen"]
mutations = ["cost", "load"]
mutations = ["load"]
topologies = ["case9", "case14"]
topologies = ["case9_30"]

# cases=["q_ext_grid","q_gen"]
for case in cases:
    for mutation in mutations:
        for topology in topologies:
            filename = f"{root_folder}/{case}/{topology}_{mutation}.json"
            df = run(filename, case)
            ax = df.plot(title=f"{topology} {mutation} mutation")
            ax.get_legend().remove()
