import sys

sys.path.append(".")
from utils.io import get_parser

from utils.logging import init_comet
from runs.homoGNN import run_case

parser = get_parser()


def run(mutations=["cost", "load_relative"], cases=["case9", "case14", "case30", "case118"],
        nb_train=8000, nb_val=2000, dataset_type="y_OPF", device="cuda", scale=0):
    for mutation in mutations:
        for case in cases:
            experiment = init_comet({"case": case, "mutation": mutation}, "test_perf_homo_v4")
            training_case = [[case, nb_train, 0.7, [mutation]]]
            validation_case = [case, nb_val, 0.7, [mutation]]
            path = "./output/test_homo_perf/" + mutation + "/" + case
            run_case(training_cases=training_case, validation_case=validation_case, plot=False,
                     title="Test performance on " + mutation, save_path=path, dataset_type=dataset_type,
                     experiment=experiment, train_batch_size=128, val_batch_size=256, scale=scale,
                     device=device)


if __name__ == "__main__":
    args = parser.parse_args()
    mutations = args.mutations.split("+")
    cases = args.cases.split("+")
    run(mutations, cases, args.nb_train, args.nb_val, device=args.device, scale=args.scale)
