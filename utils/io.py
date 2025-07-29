import json
import argparse


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_json(orient='records')
        return json.JSONEncoder.default(self, obj)


def get_parser():
    batch_train = 512
    nb_train = 8000
    nb_val = 2000
    epochs = 500
    num_workers = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mutations',
                        help="Mutations separated by +",
                        default="load_relative",
                        type=str)

    parser.add_argument('-c', '--cases',
                        help="Cases separated by +",
                        default="case9",
                        type=str)

    parser.add_argument('-vm', '--validation_mutations',
                        help="Mutations separated by +",
                        default="",
                        type=str)

    parser.add_argument('-vc', '--validation_cases',
                        help="Cases separated by +",
                        default="",
                        type=str)

    parser.add_argument('-d', '--device',
                        help="Device (cpu or cuda)",
                        default="cuda",
                        type=str)

    parser.add_argument('-sd', '--seed', help="Random seed", type=int, default=20240527)
    parser.add_argument('-nw', '--num_workers_train', help="num workers used in training", type=int,
                        default=num_workers)
    parser.add_argument('-bt', '--batch_train', help="Batch size used in training", type=int, default=batch_train)
    parser.add_argument('-t', '--nb_train', help="Number of graphs used in training", type=int, default=nb_train)
    parser.add_argument('-v', '--nb_val', help="Number of graphs used in validation", type=int, default=nb_val)
    parser.add_argument('-s', '--scale', help="Scaling features", type=int, default=0)

    parser.add_argument('-dt', '--dataset_type', help="Which features to log", type=str, default="y_OPF")
    parser.add_argument('-n', '--comet_name', help="Name of the comet project", type=str, default="")
    parser.add_argument('-o', '--opf',
                        help="simulator, possible values: 1/pandapower_opf; 2/powermodel_opf; 3/matpower_opf; pandapower_pf; Partial: opendss_pf; Partial: matpower_pf",
                        type=str, default="1")
    parser.add_argument('-r', '--ray', help="Parallelize with ray", type=int, default=0)
    parser.add_argument('-cb', '--clamp_boundary',
                        help="Clamping output; 1 clamp training only, 2 clamp both training and evaluation, 3 clamp validation only",
                        type=int,
                        default=1)
    parser.add_argument('-pl', '--use_physical_loss',
                        help="Whether to include physical loss; 1_x or  report it only, 2_x report it and minimize it with other losses, 3_x minimize it alone. 'x' version of the physic loss ",
                        type=str,
                        default="0_1")

    parser.add_argument('-he', '--hetero', help="whether to use hetero gnn representation", type=int, default=1)
    parser.add_argument('-e', '--epochs', help="Max epochs", type=int, default=epochs)
    parser.add_argument('-hp', '--num_samples', help="Num samples in hyper-param optimization", type=int, default=250)

    parser.add_argument('-cv', '--cv_ratio', help="cross validation ratio; 0 means no CV", type=float, default=0.0)

    parser.add_argument('-lr', '--base_lr', help="Initial learning rate", type=float, default=0.001)
    parser.add_argument('-dlr', '--decay_lr', help="Learnng rate decay", type=float, default=0.5)
    parser.add_argument('-cl', '--cls', help="Graph layer classes", type=str, default="sage")
    parser.add_argument('-ag', '--aggr', help="Message passing aggregation", type=str, default="mean")
    parser.add_argument('-hc', '--hidden_channels', help="Hidden layers features separated by :", type=str,
                        default="64:64")
    parser.add_argument('-we', '--weighting', help="Weighting strategy: uniform, relative, adaptive", type=str,
                        default="relative")
    parser.add_argument('-uid', '--uniqueid', help="Unique id of pre-processed dataset", type=str,
                        default="")
    parser.add_argument('-mdl', '--model_file', help="Unique id of pre-processed dataset", type=str,
                        default="")


    return parser
