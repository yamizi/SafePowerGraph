import sys

sys.path.append(".")
from utils.io import get_parser

from utils.logging import init_comet
from runs.debug_train import run_case

parser = get_parser()


def run(mutations=["cost", "load_relative"], cases=["case9", "case14", "case30", "case118"],
        nb_train=8000, nb_val=2000, dataset_type="y_no_OPF", device="cuda"):
    for mutation_src in mutations:
        for mutation_trgt in mutations:
            if mutation_src == mutation_trgt:
                continue
            for case in cases:
                experiment = init_comet({"case": case, "mutation": mutation_trgt, "mutation_src": mutation_src},
                                        "test_gen_mutation_v3")
                training_case = [[case, nb_train, 0.7, [mutation_src]]]
                validation_case = [case, nb_val, 0.7, [mutation_trgt]]
                path = "./output/test_gen_muatation/" + mutation_src + "-" + mutation_trgt + "/" + case
                run_case(training_cases=training_case, validation_case=validation_case, plot=False,
                         title="Gen performance from " + mutation_src + " to " + mutation_trgt, save_path=path,
                         dataset_type=dataset_type, experiment=experiment, train_batch_size=128, val_batch_size=256,
                         scale=False, device=device)


if __name__ == "__main__":
    args = parser.parse_args()
    mutations = args.mutations.split("+")
    cases = args.cases.split("+")
    run(mutations, cases, args.nb_train, args.nb_val, device=args.device)
