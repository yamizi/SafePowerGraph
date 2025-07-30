import sys

sys.path.append(".")
from utils.io import get_parser

from utils.logging import init_comet
from runs.fine_tuning import run_case

parser = get_parser()
parser.add_argument('-n', '--mutations_val',
                    help="Mutations separated by +",
                    default="cost+load_relative",
                    type=str)
parser.add_argument('-o', '--origin_cases',
                    help="Cases separated by +",
                    default="case9+case14+case9_case14+case30",
                    type=str)


def run(mutations=["cost", "load_relative"], origin_mutations=["cost", "load_relative"],
        origin_cases=["case9", "case14", "case30", "case118"], cases=["case9", "case14", "case30", "case118"],
        nb_train=8000, nb_val=2000, dataset_type="y_OPF", device="cuda", scale=0):
    for i, case in enumerate(cases):
        origin_case = origin_cases[i]
        mutation = mutations[i]
        origin_mutation = origin_mutations[i]
        experiment = init_comet(
            {"origin_case": origin_case, "origin_mutation": origin_mutation, "case": case, "mutation": mutation},
            "test_finetuning_v4")
        training_case = [[origin_case, nb_train, 0.7, [origin_mutation]], [case, nb_train // 10, 0.7, [mutation]]]
        validation_case = [case, nb_val, 0.7, [mutation]]
        path = "./output/test_finetuning/" + mutation + "/" + case
        run_case(training_cases=training_case, validation_case=validation_case, plot=False,
                 title="Test performance on " + mutation, save_path=path, dataset_type=dataset_type,
                 experiment=experiment, train_batch_size=128, val_batch_size=256, scale=scale,
                 device=device)


if __name__ == "__main__":
    args = parser.parse_args()
    mutations = args.mutations.split("+")
    mutations_val = args.mutations_val.split("+")
    cases = args.cases.split("+")
    origin_cases = args.origin_cases.split("+")

    cases = ["case9", "case9", "case14", "case14"] + ["case9", "case9", "case14", "case14"]
    origin_cases = ["case9", "case9", "case14", "case14"] + ["case14", "case14", "case9", "case9"]
    origin_mutations = ["load_relative", "cost", "load_relative", "cost"] + ["cost", "load", "cost", "load"]
    mutations = ["cost", "load_relative", "cost", "load_relative"] + ["cost", "load", "cost", "load"]

    origin_cases = ["case14", "case14", "case9", "case9"]
    cases = ["case9", "case9", "case14", "case14"]
    origin_mutations = ["cost", "load_relative", "cost", "load_relative"]
    mutations = ["cost", "load_relative", "cost", "load_relative"]

    start = 3
    cases = cases[start:start + 1]
    origin_cases = origin_cases[start:start + 1]
    origin_mutations = origin_mutations[start:start + 1]
    mutations = mutations[start:start + 1]

    origin_cases = ["case30"]
    cases = ["case9"]
    origin_mutations = mutations = ["load_relative"]
    run(mutations, origin_mutations, origin_cases, cases, device=args.device, scale=args.scale)
