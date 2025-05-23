import sys

sys.path.append(".")
from utils.io import get_parser
import hashlib
from utils.logging import init_comet
from runs.cases import run_case

parser = get_parser()
from runs.analyze_lines import run_case


if __name__ == "__main__":
    args = parser.parse_args()
    mutations = args.mutations.split("+")
    cases = args.cases.split("+")
    validation_mutations = args.validation_mutations.split("+")
    validation_cases = args.validation_cases.split("+") if len(args.validation_cases) else cases
    comet_name = args.comet_name if args.comet_name != "" else "test_outages_default"

    path = "./output/test_outages/" + comet_name

    mutation = mutations[0]
    training_case = [[cases[0], args.nb_train, 0.7, mutations[0]]]
    validation_case = [validation_cases[0], args.nb_val, 0.7, validation_cases[0]]

    model_file = args.model_file if args.model_file != "" else "./output/physics/model_f1f6ede6db2adfa549b7e9bf228cf8a6_1_15a951f4137adc566df614ab838cab0e.pt"
    dataset_type = args.dataset_type
    opf = args.opf
    clamp_boundary = args.clamp_boundary
    use_physical_loss = args.use_physical_loss
    weighting = args.weighting
    num_workers_train = args.num_workers_train
    base_lr = args.base_lr
    aggr = args.aggr

    experiment = init_comet({"cases": cases, "mutation": mutation},project_name=comet_name)
    hash_path = f"{training_case}_{validation_case}"
    hash_path = hashlib.md5(hash_path.encode()).hexdigest()

    run_case(training_cases=training_case, validation_case=validation_case,
             save_path=path, aggr=aggr,
             y_nodes=["gen", "ext_grid", "bus"],
             experiment=experiment, dataset_type=dataset_type,
             scale=False, opf=opf, use_ray=False, uniqueid=hash_path,
             clamp_boundary=clamp_boundary, use_physical_loss=use_physical_loss,
             weighting=weighting, num_workers_train=num_workers_train, base_lr=base_lr, model_file=model_file)
