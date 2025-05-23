from pandapower.topology.create_graph import create_nxgraph
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import torch
from utils.io import JSONEncoder
from typing import Callable, Optional
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset
)
from pandapower.auxiliary import pandapowerNet
# from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import json
from copy import deepcopy
from utils.pandapower.normalization import normalizeCols, MAX_ANGLE
import torch_geometric.transforms as T

INF_VAL = 10 ** 5


class PandaPowerGraph(InMemoryDataset):
    def __init__(self, network: pandapowerNet, preprocess: Optional[str] = None,
                 transform: Optional[Callable] = None, scale=True,
                 pre_transform: Optional[Callable] = None, device="cpu",
                 include_res: bool = True, opf_as_y: bool = True, hetero=True,
                 y_nodes=["gen", "ext_grid", "bus", "line"]):

        transform =  T.Compose([T.ToUndirected(merge=True), T.ToDevice(device)]) if transform is None else transform
        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        assert self.preprocess in [None, 'metapath2vec', 'transe']
        super().__init__(None, transform, pre_transform)

        self.device = device
        self.node_types = ["bus", "load", "shunt", "ext_grid", "gen", "sgen", "line", "trafo", "trafo3w", "impedance",
                           "xward"]
        self.y_nodes = y_nodes
        self.network = network
        self.include_res = include_res
        self.opf_as_y = opf_as_y
        self.scale = scale

        if hetero:
            hetero_data, edges, dataframes, scalers = self.build_hetero_data(network, include_res, opf_as_y,
                                                                             scale=scale)
        else:
            homo_data, dataframes, scalers = self.build_homo_data(network, include_res, opf_as_y, scale=scale)


    def set_device(self,device, num_workers):
        self.data.to("cpu")
        if num_workers>0:
            device="cpu"
        else:
            self.to(device)
        self.transform = T.Compose([T.ToUndirected(merge=True), T.ToDevice(device)])
        return self

    def to(self, device):
        self.data = self.data.to(device)
        return self

    def num_features(self, node):
        default_features = {"bus": 12, "load": 10, "shunt": 10, "ext_grid": 15, "gen": 18, "sgen": 7, "line": 27,
                            "trafo": 29, "trafo3w": 35, "impedance": 8, "xward": 10}
        return default_features.get(node, 1)

    @property
    def num_outputs(self) -> int:
        return len(
            self.output_nodes)  # ["p_mw", "q_mvar", "vm_pu", "va_degree", "pl_mw", "ql_mvar","i_from_ka", "i_to_ka"]

    @property
    def output_nodes(self) -> [str]:
        nodes_features = {"bus": ["vm_pu", "va_degree"], "gen": ["p_mw", "q_mvar"], "ext_grid": ["p_mw", "q_mvar"]}
        return [a for e in self.y_nodes for a in nodes_features.get(e, [])]

    @property
    def total_output_nodes(self) -> [str]:
        return np.sum([len(self.dataframes.get(e)) for e in self.y_nodes]) * 2

    def export(self, filename="export", format="json", experiment=None):

        if format == "csv":
            for (sheetname, sheet) in self.dataframes.items():
                sheet.to_csv(filename + "_" + sheetname + ".csv")
        elif format == "json":
            with open(filename + "." + format, "w") as f:
                json.dump(self.dataframes, f, cls=JSONEncoder)
            if experiment is not None:
                experiment.log_asset(filename + "." + format, filename + "." + format)
        elif format == "xlsx":
            with open(filename + "." + format, "wb") as f:
                for (sheetname, sheet) in self.dataframes.items():
                    sheet.to_excel(f, sheet_name=sheetname)

    def build_homo_data(self, network, include_res=True, opf_as_y=True, scale=True):
        node = "bus"
        bus_df = deepcopy(getattr(network, node)) if (
                len(getattr(network, "res_" + node)) == 0 or not include_res) else pd.merge(getattr(network, node),
                                                                                            getattr(network,
                                                                                                    "res_" + node),
                                                                                            "left", on=None,
                                                                                            left_index=True,
                                                                                            right_index=True)
        bus_df.drop(columns=["name"], inplace=True)
        bus_df["n_id"] = bus_df.index
        merged_bus_df = bus_df.copy(True)

        node = "ext_grid"
        ext_grid_df = deepcopy(getattr(network, node)) if (
                len(getattr(network, "res_" + node)) == 0 or not include_res) else pd.merge(getattr(network, node),
                                                                                            getattr(network,
                                                                                                    "res_" + node),
                                                                                            "left", on=None,
                                                                                            left_index=True,
                                                                                            right_index=True)
        ext_grid_df["has_grid"] = 1
        ext_grid_df.drop(columns=["name"], inplace=True)
        if len(ext_grid_df):
            merged_bus_df = pd.merge(merged_bus_df, ext_grid_df, left_on="n_id", right_on="bus", how="left",
                                     suffixes=("", "_ext_grid"))

        node = "gen"
        gen_df = deepcopy(getattr(network, node)) if (
                len(getattr(network, "res_" + node)) == 0 or not include_res) else pd.merge(getattr(network, node),
                                                                                            getattr(network,
                                                                                                    "res_" + node),
                                                                                            "left", on=None,
                                                                                            left_index=True,
                                                                                            right_index=True)
        gen_df["has_gen"] = 1
        gen_df.drop(columns=["name"], inplace=True)
        if len(gen_df):
            merged_bus_df = pd.merge(merged_bus_df, gen_df, left_on="n_id", right_on="bus", how="left",
                                     suffixes=("", "_gen"))

        node = "sgen"
        sgen_df = deepcopy(getattr(network, node)) if (
                len(getattr(network, "res_" + node)) == 0 or not include_res) else pd.merge(getattr(network, node),
                                                                                            getattr(network,
                                                                                                    "res_" + node),
                                                                                            "left", on=None,
                                                                                            left_index=True,
                                                                                            right_index=True)
        sgen_df["has_sgen"] = 1
        sgen_df.drop(columns=["name"], inplace=True)
        if len(sgen_df):
            merged_bus_df = pd.merge(merged_bus_df, sgen_df, left_on="n_id", right_on="bus", how="left",
                                     suffixes=("", "_sgen"))

        merged_bus_df = merged_bus_df.fillna(0)
        cols = [a for a in merged_bus_df.columns if
                (("q_mvar" in a) or ("p_mw" in a)) and not ("min" in a or "max" in a)]
        x = merged_bus_df.drop(columns=cols)

        scaler = None  # StandardScaler()
        one_hot = pd.get_dummies(x).dropna(axis=1).values.astype("float32")
        if scale and scaler is not None:
            one_hot = scaler.fit_transform(one_hot)
        # x_dict = dict(zip(range(len(one_hot)), one_hot.tolist()))
        x_dict = dict(zip(range(len(one_hot)), torch.Tensor(one_hot).to(self.device)))
        y_gen = np.concatenate(
            [pd.merge(getattr(network, node)[["bus"]], getattr(network, "res_" + node), "left", on=None,
                      left_index=True, right_index=True)[["bus", "p_mw", "q_mvar"]].values for node in
             ["ext_grid", "gen", "sgen"]])

        y_bus = getattr(network, "res_bus")[["vm_pu", "va_degree"]].values
        y = np.ones((y_bus.shape[0], 6)) * np.nan
        y[:, 2:4] = y_bus
        y[y_gen[:, 0].astype(int), 0:2] = y_gen[:, 1:3]
        y[y_gen[:, 0].astype(int), 2:4] = np.nan

        y_dict = dict(zip(range(len(y)), torch.Tensor(y).to(self.device)))
        y_dict_default = dict(
            zip(list(range(len(bus_df))), [torch.Tensor([np.nan, np.nan]).to(self.device)] * len(bus_df)))
        nxgraph = create_nxgraph(network, multi=False, calc_branch_impedances=True)
        nx.set_node_attributes(nxgraph, x_dict, "x")
        nx.set_node_attributes(nxgraph, y_dict, "y")
        graph = from_networkx(nxgraph)

        dataframes = {"bus": x, "ext_grid": ext_grid_df, "gen": gen_df, "sgen": sgen_df}

        scalers = {"bus": scaler}
        self.data, self.slices = graph, None
        self.scalers = scalers
        self.dataframes = dataframes

        return graph, dataframes, scalers

    def build_hetero_data(self, network, include_res=True, opf_as_y=True, scale=False, boundary_tolerance=1e-5):

        node_types = self.node_types
        costs = network.poly_cost
        data = HeteroData()
        edges = {}
        dataframes = {}
        scalers = {}

        bus_index = {e: k for (k, e) in enumerate(network.bus.index.to_list())}

        for node in node_types:
            edges_ = []
            edges_from = []
            edges_to = []
            edges_to2 = []

            if not include_res or len(getattr(network, "res_" + node)) == 0 or not node in self.y_nodes:
                merged_df = deepcopy(getattr(network, node))
            else:
                merged_df = pd.merge(getattr(network, node), getattr(network, "res_" + node), "left", on=None,
                                     left_index=True,
                                     right_index=True)

            if len(merged_df):
                boundaries = np.nan * np.ones((len(merged_df), self.num_outputs * 2))  # to support min and max values
                mask = np.zeros((len(merged_df), self.num_outputs))
                # merged_df = normalizeCols(merged_df, columns=["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"],
                #                           min_val=-network.sn_mva, max_val=network.sn_mva)

                if node == "ext_grid" or node == "gen":
                    y = ["p_mw", "q_mvar"]
                    if node == "gen":
                        start_index = 0
                    else:
                        start_index = 1
                    mask[:, start_index * 2:(start_index + 1) * 2] = 1
                    drop_y = y
                    # we enforce boundaries slightly tighter than original boundaries in the loss estimation
                    boundaries_conservative = merged_df[["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]].values
                    boundaries_conservative = boundaries_conservative + np.repeat([[10 * boundary_tolerance,
                                                                                    10 * boundary_tolerance,
                                                                                    -10 * boundary_tolerance,
                                                                                    -10 * boundary_tolerance]],
                                                                                  len(merged_df), 0)
                    boundaries[:, start_index * 4:(start_index + 1) * 4] = boundaries_conservative / network.sn_mva
                elif node == "sgen":
                    y = ["p_mw", "q_mvar"]
                    drop_y = y
                elif node in ["bus"]:
                    y = ["vm_pu", "va_degree"]

                    # For Vm
                    boundaries_conservative = merged_df[["min_vm_pu", "max_vm_pu"]].values
                    boundaries_conservative = boundaries_conservative + np.repeat([[10 * boundary_tolerance,
                                                                                    -10 * boundary_tolerance]],
                                                                                  len(merged_df), 0)
                    boundaries[:, 8:10] = boundaries_conservative

                    # For Va
                    boundaries[:, 10] = -1 + 10 * boundary_tolerance
                    boundaries[:, 11] = 1 - 10 * boundary_tolerance

                    mask[:, 4:6] = 1
                    drop_y = y


                elif node in ["line"]:
                    if node in self.y_nodes:
                        mask[:, 6:10] = 1
                        max_lines = merged_df[["max_i_ka"]].values - 10 * boundary_tolerance
                        if boundaries.shape[1] >= 20:
                            boundaries[:, 16:20] = np.concatenate(
                                [np.zeros_like(max_lines), max_lines, np.zeros_like(max_lines), max_lines], 1)

                    y = ["pl_mw", "ql_mvar", "i_from_ka", "i_to_ka"]
                    drop_y = ["pl_mw", "ql_mvar", "i_from_ka", "i_to_ka", "i_ka", "vm_from_pu", "va_from_degree",
                              "vm_to_pu", "va_to_degree", "loading_percent","p_from_mw",'q_from_mvar', 'p_to_mw',
                            'q_to_mvar', 'df']
                data[node].boundaries = torch.Tensor(boundaries).to(self.device)
                data[node].output_mask = torch.Tensor(mask).to(self.device)
            if opf_as_y:

                if node in self.y_nodes and len(getattr(network, "res_" + node)) > 0:
                    merged_df = merged_df.rename(columns={"p_mw_y": "p_mw", "vm_pu_y": "vm_pu", 'q_mvar_y': 'q_mvar'})
                    if include_res:
                        merged_df.drop(columns=drop_y, inplace=True)
                    pf = getattr(network, "res_" + node)

                    # Convert angles to radian to normalize angles
                    if "va_degree" in pf.columns:
                        pf["va_degree"] = np.deg2rad(pf["va_degree"])

                    values = torch.Tensor(pf[y].values).to(self.device)
                    # Normalize P & Q with nominal voltage
                    if y[0] == "p_mw":
                        values[:, 0] = values[:, 0] / network.sn_mva
                    if y[1] == "q_mvar":
                        values[:, 1] = values[:, 1] / network.sn_mva
                    data[node].y = values

                    if node in ["ext_grid", "gen"]:
                        node_cost = costs[costs["et"] == node]
                        node_cost.index = node_cost.element
                        merged_df = pd.merge(merged_df, node_cost, how="left", right_index=True, left_index=True).drop(
                            columns=["et", "element"])

            merged_df.drop(columns=["name"], inplace=True)
            if node == "bus":
                merged_df.drop(columns=["type", "zone"], inplace=True)

            scaler = None  # StandardScaler()
            merged_df = merged_df.replace([np.inf, -np.inf], [-INF_VAL, INF_VAL]).dropna(axis=1)
            # normalize PQ
            # merged_df = normalizeCols(merged_df, columns=["p_mw", "q_mvar"], min_val=-network.sn_mva, max_val=network.sn_mva)

            one_hot = pd.get_dummies(merged_df).values.astype("float32")
            if scale and scaler is not None:
                one_hot = scaler.fit_transform(one_hot)

            data[node].x = torch.Tensor(one_hot).to(self.device) if len(one_hot) else torch.zeros(1, self.num_features(
                node)).to(self.device)
            scalers[node] = scaler

            if "from_bus" in merged_df.columns:
                edges_from = [bus_index[e] for e in merged_df["from_bus"].values]
                merged_df.drop(columns=["from_bus"], inplace=True)
                data['bus', 'to', node].edge_index = torch.LongTensor([edges_from, merged_df.index.tolist()]).to(
                    self.device)

            if "to_bus" in merged_df.columns:
                edges_to = [bus_index[e] for e in merged_df["to_bus"].values]
                merged_df.drop(columns=["to_bus"], inplace=True)
                data[node, 'to', 'bus'].edge_index = torch.LongTensor([merged_df.index.tolist(), edges_to]).to(
                    self.device)

            if "hv_bus" in merged_df.columns:
                edges_from = [bus_index[e] for e in merged_df["hv_bus"].values]
                merged_df.drop(columns=["hv_bus"], inplace=True)
                data['bus', 'to', node].edge_index = torch.LongTensor([edges_from, merged_df.index.tolist()]).to(
                    self.device)

                edges_to = [bus_index[e] for e in merged_df["lv_bus"].values]
                merged_df.drop(columns=["lv_bus"], inplace=True)
                data[node, 'to', 'bus'].edge_index = torch.LongTensor([merged_df.index.tolist(), edges_to]).to(
                    self.device)

            if "mv_bus" in merged_df.columns:
                edges_to2 = [bus_index[e] for e in merged_df["mv_bus"].values]
                merged_df.drop(columns=["mv_bus"], inplace=True)
                data[node, 'to', 'bus'].edge_index = torch.LongTensor(
                    [merged_df.index.tolist() + merged_df.index.tolist(), edges_to + edges_to2]).to(self.device)

            if "bus" in merged_df.columns:
                edges_ = [bus_index[e] for e in merged_df["bus"].values]
                merged_df.drop(columns=["bus"], inplace=True)
                data['bus', 'to', node].edge_index = torch.LongTensor([edges_, merged_df.index.tolist()]).to(
                    self.device)

            dataframes[node] = merged_df
            edges[node] = [edges_, edges_from, edges_to, edges_to2]

            # node, len(getattr(network,node).columns), len(merged_df.columns))

        for k in dataframes.keys():
            assert len(torch.isnan(data[k].x).int().nonzero()) == 0
        for k in self.y_nodes:  # dataframes.keys():
            assert len(torch.isnan(data[k].y).int().nonzero()) == 0 if hasattr(data[k], "y") else True

        data.sn_mva = network.sn_mva
        data.angle = "rad"
        data.f_hz = network.f_hz

        self.data, self.slices = data, None
        self.scalers = scalers
        self.dataframes = dataframes
        self.edges = edges

        return data, edges, dataframes, scalers
