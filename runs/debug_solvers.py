import os.path
import sys
import hashlib

sys.path.append(".")
from utils.logging import init_comet
import hashlib
from runs.analyze_solvers import run_case

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    max_epochs = 0

    cases=["case9"]#    ,"case30","case118"]
    mutations = ["line_nminus1:1:1","cost","load_relative"]
    mutations = ["load_relative"]

    model_file = ""

    for case in cases:
        for mutation in mutations:
            training_case = [[case, 1, 0.7, [mutation]]]
            validation_case = [case, 5, 0.7, [mutation]]

            opf = "pandapower_opf+powermodel_opf"
            ref_pf = "matpower_opf"
            project_name = "debug_solvers_opf"

            ref_pf = "matpower_pf"
            opf="pandapower_pf+opendss_pf"
            opf = "pandapower_pf"
            project_name = "debug_solvers_pf"

            experiment = init_comet({"case": case, "mutation": mutation},project_name="debug_solvers_pf")
            hash_path = f"{training_case}_{validation_case}"
            hash_path = hashlib.md5(hash_path.encode()).hexdigest()
            run_case(training_cases=training_case, validation_case=validation_case,
                     model_file=model_file,save_path=f"./output/"+project_name,
                     y_nodes=["gen", "ext_grid", "bus"],ref_pf=ref_pf,
                      experiment=experiment, dataset_type="y_OPF", seed=100,
                     scale=False,opf=opf, use_ray=False, uniqueid=hash_path)
