import sys

sys.path.append(".")
from utils.io import get_parser
from experiments.test_performances import run
parser = get_parser()




if __name__ == "__main__":
    args = parser.parse_args()
    mutations = args.mutations.split("+")
    cases = args.cases.split("+")
    validation_mutations = args.validation_mutations.split("+")
    validation_cases = args.validation_cases.split("+")
    hidden_channels = args.hidden_channels.split(":")
    comet_name = args.comet_name if args.comet_name != "" else "test_perf_a1"
    run(mutations, cases, args.nb_train, args.nb_val, device=args.device, scale=args.scale, epochs=0,
        dataset_type=args.dataset_type, opf=args.opf, project_name=comet_name, use_ray=args.ray,
        num_samples=args.num_samples, cv_ratio=args.cv_ratio, aggr=args.aggr, cls=args.cls, decay_lr=args.decay_lr,
        base_lr=args.base_lr, hidden_channels=hidden_channels, batch_train=args.batch_train,
        clamp_boundary=args.clamp_boundary, use_physical_loss=args.use_physical_loss, weighting=args.weighting,
        uniqueid=args.uniqueid, seed=args.seed, validation_cases=validation_cases,
        validation_mutations=validation_mutations, build_db_only=True)
