import os.path
import sys
import hashlib

sys.path.append(".")
from utils.logging import init_comet
import hashlib
from runs.analyze_constraints import run_case

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    best_models = {
        "case9":("gat","./output/test_perf/perf_opf_weightingV2/model_9887f94c4121308cf9b9d2aa62c76fce_20240620_a4214eb7d614d771471296c63bf7e9ef.pt"),
        "case30": ("gat",
                   "./output/test_perf/perf_opf_weightingV2/model_cd692877a28c9c5ec6a847523b82f937_20240620_68f487d56ceaa347a3e4cb3324e73123.pt")
    }


    max_epochs = 0
    case = "case30"
    mutation="load_relative"

    opf = 1
    cv_ratio = 0
    clamp_boundary = 0
    use_physical_loss = "2_2"
    weighting = "uniform"
    num_workers_train = 2
    base_lr = 0.01
    hidden_channels=[128,128]

    cases = ["case30","case9"] #, "case118"
    mutations = ["cost","load_relative","line_nminus1:1:1"]

    for case in cases:
        for mutation in mutations:
            training_case = [[case, 100, 0.7, [mutation]]]
            validation_case = [case, 100, 0.7, [mutation]]

            experiment = init_comet({"case": case, "mutation": mutation},project_name="debug_constraints")
            hash_path = f"{training_case}_{validation_case}"
            hash_path = hashlib.md5(hash_path.encode()).hexdigest()

            #best case30 https://www.comet.com/hgnn/perf-opf-weightingv2/2f6dc782d88c4a10a3d6788b56719b3c
            cls, model_file = best_models.get(case)
            #model_file = "./output/physics/model_91eaece9d52d367b29f843b27871ab80_1_16a4caea46f105e47294572449a2360c.pt"

            run_case(training_cases=training_case, validation_case=validation_case,
                    save_path=f"./output/constraints",cls=cls,hidden_channels=hidden_channels,
                     y_nodes=["gen", "ext_grid", "bus"],
                      experiment=experiment, dataset_type="y_OPF",
                     scale=False,opf=opf, use_ray=True, uniqueid=hash_path,
                     clamp_boundary=clamp_boundary, use_physical_loss=use_physical_loss,
                     weighting=weighting, num_workers_train=num_workers_train, base_lr=base_lr, model_file=model_file)
