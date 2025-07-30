import sys
import os
sys.path.append(".")
from utils.io import get_parser
import hashlib
from utils.logging import init_comet
from runs.cases import run_case

parser = get_parser()


def run(mutations=["cost", "load_relative"], cases=["case9", "case14", "case30", "case118"],
        nb_train=8000, nb_val=2000, dataset_type="y_OPF", device="cuda", scale=0, cv_ratio=0.2,
        opf=1, project_name="test_perf_v4", use_ray=1, epochs=500, num_samples=200, aggr="mean", cls="sage",
        base_lr=0.1, decay_lr=0.5, hidden_channels=[64, 64], batch_train=256, clamp_boundary=0, use_physical_loss="1_1"
        , weighting="relative", uniqueid="", seed=1, validation_mutations=[], validation_cases=[], build_db_only=False,
        num_workers_train=0, model_file="", args=None):
    # for mutation in mutations:
    #    for case in cases:

    validation_cases = cases[0] if len(validation_cases) or validation_cases[0] == "" else validation_cases
    validation_mutations = mutations[0] if len(validation_mutations)==0 or validation_mutations[
        0] == "" else validation_mutations[0]

    initial_params = {"case": "+".join(cases), "mutation": "+".join(mutations),
                      "validation_cases": validation_cases,
                      "validation_mutations": validation_mutations}
    experiment = init_comet(initial_params, project_name)

    training_cases = [[case, nb_train, 0.7, mutation.split("#")] for case in cases for mutation in mutations]
    validation_case = [validation_cases, nb_val, 0.7, validation_mutations.split("#")]

    path = os.path.join(args.project_path , project_name)

    if uniqueid == "":
        hash_path = f"{'+'.join(cases)}/{validation_cases}/{nb_train}/{'+'.join(mutations)}/{validation_mutations}/{nb_val}"
        print("path before hash", hash_path)
        hash_path = hashlib.md5(hash_path.encode()).hexdigest()
        print("path after hash", hash_path)
    else:
        hash_path = uniqueid
    # hash_path = hash(hash_path)

    run_case(training_cases=training_cases, validation_case=validation_case, plot=False,
             title="Test performance", save_path=path, dataset_type=dataset_type,
             experiment=experiment, train_batch_size=batch_train, val_batch_size=512, scale=scale,
             device=device, opf=opf, use_ray=use_ray, uniqueid=hash_path, max_epochs=epochs, cv_ratio=cv_ratio,
             num_samples=num_samples, aggr=aggr, cls=cls, base_lr=base_lr, decay_lr=decay_lr,
             hidden_channels=hidden_channels, clamp_boundary=clamp_boundary,build_db_only=build_db_only,
             use_physical_loss=use_physical_loss, weighting=weighting, seed=seed, num_workers_train=num_workers_train,
             model_file=model_file, args=args)


if __name__ == "__main__":
    args = parser.parse_args()
    mutations = args.mutations.split("+")
    cases = args.cases.split("+")
    validation_mutations = args.validation_mutations.split("+")
    validation_cases = args.validation_cases.split("+")
    hidden_channels = args.hidden_channels.split(":")
    comet_name = args.comet_name if args.comet_name != "" else "test_perf_a1"
    run(mutations, cases, args.nb_train, args.nb_val, device=args.device, scale=args.scale, epochs=args.epochs,
        dataset_type=args.dataset_type, opf=args.opf, project_name=comet_name, use_ray=args.ray,
        num_samples=args.num_samples, cv_ratio=args.cv_ratio, aggr=args.aggr, cls=args.cls, decay_lr=args.decay_lr,
        base_lr=args.base_lr, hidden_channels=hidden_channels, batch_train=args.batch_train,
        clamp_boundary=args.clamp_boundary, use_physical_loss=args.use_physical_loss, weighting=args.weighting,
        uniqueid=args.uniqueid, seed=args.seed, validation_cases=validation_cases,
        validation_mutations=validation_mutations,num_workers_train=args.num_workers_train, model_file=args.model_file,
        args=args)
