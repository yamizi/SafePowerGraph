import os.path
import sys
import hashlib

sys.path.append(".")
import uuid
from matplotlib import pyplot as plt

from utils.logging import init_comet
import hashlib
from runs.cases import run_case

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    max_epochs = 100
    case = "case1354pegase"
    case = "case9"
    mutation = "line_nminus1"
    mutation="load_relative"
    training_case = [[case, 15, 0.7, [mutation]]]
    validation_case = [case, 5, 0.7, [mutation]]
    opf = 1
    cv_ratio = 0
    clamp_boundary = 3
    use_physical_loss = "21_2"
    weighting= "relative sup2ssl"
    weighting = "uniform"
    num_workers_train = 2
    base_lr = 0.01
    filename="OPF_PF/OPF-PF_case30_load_relative_20.pkl"
    #OPF_case9_load_relative_20.pkl
    ### Add your own Huggingface token
    token=""
    ### 20/25/200
    seed=20

    experiment = init_comet({"case": case, "mutation": mutation})
    hash_path = f"{training_case}_{validation_case}"
    hash_path = hashlib.md5(hash_path.encode()).hexdigest()
    #hash_path = hash(hash_path)
    run_case(training_cases=training_case, validation_case=validation_case, val_batch_size=50, train_batch_size=5,
             title="generalization load_relative", save_path=f"./output/physics",
             y_nodes=["gen", "ext_grid", "bus"],
             max_epochs=max_epochs, experiment=experiment, dataset_type="y_OPF",
             scale=False, filter=True, opf=opf, use_ray=False, uniqueid=hash_path,
             cv_ratio=cv_ratio, clamp_boundary=clamp_boundary, use_physical_loss=use_physical_loss,
             weighting=weighting, num_workers_train=num_workers_train, base_lr=base_lr,seed=seed,
             token="")
    plt.show()
    exit()

    training_case = [["case9", 64, 0.7, ["cost"]]]
    validation_case = ["case14", 32, 0.7, ["cost"]]
    run_case(training_cases=training_case, validation_case=validation_case,
             title="generalization cost", save_path="./output/case9_14")
