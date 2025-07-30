import os.path
import sys
import hashlib
sys.path.append('/home/bizon/Projects/Max/output/matpower')
#sys.path.append(".")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logging import init_comet
import hashlib
from runs.analyze_solvers_initialization import run_case

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    max_epochs = 0
    num_initial_states = 100

    cases=["case9","case30","case118"]
    mutations = ["load_relative", "line_nminus1:1:1","cost"]
    case = "case9"
    mutation="load_relative"
    model_file = ""

    for case in cases:
        for mutation in mutations:
            training_case = [[case, 1, 0.7, [mutation]]]
            validation_case = [case, 100, 0.7, [mutation]]
            opf = "pandapower_opf"

            experiment = init_comet({"case": case, "mutation": mutation},project_name="debug_solvers_init")
            hash_path = f"{training_case}_{validation_case}"
            hash_path = hashlib.md5(hash_path.encode()).hexdigest()

            run_case(validation_case=validation_case,model_file=model_file,
                    save_path=f"./output/line_analysis",num_initial_states=num_initial_states,
                     y_nodes=["gen", "ext_grid", "bus"],
                      experiment=experiment, dataset_type="y_OPF",
                     scale=False,opf=opf, use_ray=True, uniqueid=hash_path)
