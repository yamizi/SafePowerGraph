import pandapower as pp
import copy, os
from typing import Tuple
import numpy as np
import torch
from utils.attacks.losses import AdvLoss
from evotorch import Problem, Solution
from evotorch.algorithms import SNES
from pandapower.auxiliary import pandapowerNet
from utils.pandapower.pandapower_graph import PandaPowerGraph
from utils.train import heterogeneous_loss_step
from utils.losses import node_loss
import torch_geometric.transforms as T
from evotorch.logging import StdOutLogger, PandasLogger
import ray

MUTABLE_FEATURES = {
    "load": ['p_mw', 'q_mvar'],
    "line": ['r_ohm_per_km', 'x_ohm_per_km', 'c_nf_per_km', 'g_us_per_km']
}


class AdversarialProblem(Problem):
    def __init__(self, model: torch.nn.Module, initial_network: PandaPowerGraph, mutable_features: torch.Tensor,
                 initial_data:dict=None, solution_length: int = 2, loss: torch.nn.Module = AdvLoss, opf_algorithm=1,
                 num_actors:int=1, log=True, max_bound=1.2):
        super().__init__(
            objective_sense="max",
            solution_length=solution_length,
            initial_bounds=(0, max_bound),
            num_actors=num_actors
        )

        # Store the A parameter for evaluation
        self._model = ray.put(model)
        self._loss = loss
        self._initial_network = ray.put(initial_network)
        self._mutable_features = mutable_features
        self._opf_algorithm = opf_algorithm
        self._log = log
        self._y_nodes = ["gen", "ext_grid", "bus"]
        self._initial_data = initial_data

        data = T.ToUndirected(merge=True)(initial_network.data)
        model_out = model(data.x_dict, data.edge_index_dict)


    def run_opf(self, network):
        opf = self._opf_algorithm
        diagnostic_errors = pp.diagnostic(copy.deepcopy(network), report_style="compact")
        if len(diagnostic_errors)>0:
            network = None
            msg = diagnostic_errors.__str__()
        else:
            msg = ""
            try:
                if int(opf) == 1 or opf == "pandapower_opf":
                    pp.runopp(network)

                elif int(opf) == 2 or opf=="powermodel_opf":
                    pp.runpm_ac_opf(network)

                elif int(opf) == 3 or opf=="matpower_opf":
                    octave_path = os.environ.get("OCTAVE_PATH", None)
                    from runs.matpower import opf as matpower_opf

                    case = network.case
                    networks, _ = matpower_opf(case=case, loads=None, lines_in_service=None,
                                                             octave_path=octave_path)
                    network = network[0]
            except pp.OPFNotConverged as e:
                network = None

        if not network or not network.OPF_converged:
            if self._log:
                print("not converged opf", msg)
            return None

        return network

    def mutate_graph(self,x):
        network = ray.get(self._initial_network)
        pandapower_network = copy.deepcopy(network.network)
        initial_network = copy.deepcopy(network)

        start_features = 0
        return initial_network, pandapower_network

        for node, features in self._mutable_features.items():
            features_multiplier = x[start_features:start_features + len(features[0]) * features[1]]
            pandapower_network.get(node)[features[0]] = pandapower_network.get(node)[
                                     features[0]] * features_multiplier.reshape(-1, len(features[0])).cpu().numpy()

        return initial_network, self.run_opf(pandapower_network)
    def _evaluate(self, solution: Solution):
        x = solution.values
        initial_network, network = self.mutate_graph(x)
        model = ray.get(self._model)

        if network is None:
            output = -torch.inf
        else:
            data, edges, dataframes, scalers = initial_network.build_hetero_data(network,
                                                                                 initial_network.include_res,
                                                                                 initial_network.opf_as_y,
                                                                                 scale=initial_network.scale)
            data = T.ToUndirected(merge=True)(data)
            model_out = model(data.x_dict, data.edge_index_dict)
            loss = 0
            losses = []
            boundary_losses = []
            physical_losses = []
            cost_losses = []

            return_label = {}
            return_output = {}

            use_physical_loss = "0"
            with torch.no_grad():
                return_output, return_label, loss, losses, boundary_losses, physical_losses, cost_losses = heterogeneous_loss_step(
                    self._y_nodes, data, model_out, loss, node_loss, True, return_output, return_label,
                    losses, boundary_losses, use_physical_loss, physical_losses, cost_losses, None)
            output = loss

        solution.set_evals(output)


class SearchAttack(torch.nn.Module):
    search_params = {

    }

    def __init__(
            self,
            model: torch.nn.Module,
            loss: torch.nn.Module = AdvLoss,
            metric: torch.nn.Module = None,
            algorithm: str = "SNES",
            algo_params: dict = {},
            log: bool = True,
            **kwargs,
    ):
        super().__init__()

        self.model = model

        self.loss = loss
        self.algorithm = algorithm
        self.algo_params = algo_params

        self.log = log
        self.metric = metric or self.loss

        self.search_params.update(kwargs)

    def attack(
            self,
            initial_network: PandaPowerGraph,
            budget: int = 100,
            data:dict=None,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attack the predictions for the provided model and graph.

        A subset of predictions may be specified with :attr:`idx_attack`. The
        attack is allowed to flip (i.e. add or delete) :attr:`budget` edges and
        will return the strongest perturbation it can find. It returns both the
        resulting perturbed :attr:`edge_index` as well as the perturbations.

        Args:
            initial_graph (torch.Tensor): The labels.
            budget (int): The number of search iterations.

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)
        """

        mutable_nodes = self.algo_params.get("mutable_nodes", "load").split("#")
        mutable_features = {n: (MUTABLE_FEATURES.get(n), len(initial_network.network.get(n))) for n in mutable_nodes}
        nb_mutable_features = np.sum([len(e[0]) * e[1] for e in mutable_features.values()])

        self.model.eval()
        num_actors = kwargs.get("num_actors",1)
        problem = AdversarialProblem(self.model, initial_network, initial_data=data, mutable_features=mutable_features,
                                     solution_length=nb_mutable_features, loss=self.loss, log=self.log, num_actors=num_actors)

        searcher = None
        if self.algorithm == "SNES":
            searcher = SNES(problem, popsize=kwargs.get("popsize",100), stdev_init=0.01)

        stdout_logger = StdOutLogger(searcher)
        pandas_logger = PandasLogger(searcher)
        searcher.run(budget)
        best_discovered_solution = searcher.status["best"]

        progress = pandas_logger.to_dataframe()
        #progress.mean_eval.plot()


        return None, best_discovered_solution
