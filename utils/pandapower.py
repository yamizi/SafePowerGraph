from typing import Callable, Optional
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset
)
from pandapower.auxiliary import pandapowerNet
import pandapower as pp
import numpy as np
import torch
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

def build_costs(net, costs):
     #net.gen["cost"] = 0
     #net.ext_grid["cost"] = 0
     #net.sgen["cost"] = 0

     for cost in costs:
        pp.create_poly_cost(net, cost[1], cost[0], cp1_eur_per_mw=cost[2], check=False)
        #getattr(net,cost[0]).at[cost[1],'cost']=cost[2]


class PandaPowerDataset(InMemoryDataset):
    def __init__(self, network: pandapowerNet, preprocess: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 include_res:bool=True, opf_as_y:bool=True):
        
        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        assert self.preprocess in [None, 'metapath2vec', 'transe']
        super().__init__(None, transform, pre_transform)

        hetero_data, edges, dataframes, scalers= build_hetero_data(network, include_res, opf_as_y)
        self.data, self.slices = hetero_data, None
        self.scalers = scalers
        self.dataframes = dataframes
        self.edges = edges

    @property
    def num_outputs(self) -> int:
        return 2 # np.sum([len(self._data[e].y.flatten()) for e in self.output_nodes if hasattr(self._data[e],"y")])
    
    @property
    def output_nodes(self) -> [str]:
        return [e for e in ["ext_grid","sgen","gen"] if hasattr(self._data[e],"y")]

def build_torch_dataset(network):
    hetero_data, edges, dataframes= build_hetero_data(network)

def build_hetero_data(network, include_res=True, opf_as_y=True):
     
    node_types = ["bus","load","sgen","gen","shunt","ext_grid","line","trafo","trafo3w","impedance","xward"]
    data = HeteroData()
    edges = {}
    dataframes = {}
    scalers = {}
    for node in node_types:
        edges_ = []
        edges_from = []
        edges_to = []
        edges_to2 = []

        merged_df = deepcopy(getattr(network,node)) if (len(getattr(network,"res_"+node))==0 or not include_res) else pd.merge(getattr(network,node),getattr(network,"res_"+node),"left",on=None,left_index=True,right_index=True)
        if len(merged_df)==0:
            continue
        if opf_as_y and node in ["gen","sgen","ext_grid"] and len(getattr(network,"res_"+node))>0:
            y = ["p_mw","q_mvar"]
            if include_res:
                if node=="ext_grid":
                    merged_df.drop(columns=["p_mw","q_mvar"],inplace=True)
                if node in ["sgen","gen"]:
                    merged_df.drop(columns=["va_degree","vm_pu_y", "p_mw_y","q_mvar"],inplace=True)
                    # y = ["p_mw","q_mvar", "va_degree"]
            pf = getattr(network,"res_"+node)[y]
            data[node].y =torch.Tensor(pf.values.tolist())

        merged_df.drop(columns=["name"],inplace=True)   
        scaler = StandardScaler()
        data[node].x =torch.Tensor(scaler.fit_transform(pd.get_dummies(merged_df).dropna(axis=1)))
        scalers[node] = scaler

        if "from_bus" in merged_df.columns:
            edges_from = merged_df["from_bus"].tolist()
            merged_df.drop(columns=["from_bus"],inplace=True)
            data['bus','to',node].edge_index = torch.LongTensor([edges_from, merged_df.index.tolist()])
 
        if "to_bus" in merged_df.columns:
            edges_to = merged_df["to_bus"].tolist()
            merged_df.drop(columns=["to_bus"],inplace=True)
            data[node,'to','bus'].edge_index = torch.LongTensor([merged_df.index.tolist(),edges_to])

        if "hv_bus" in merged_df.columns:
            edges_from = merged_df["hv_bus"].tolist()
            merged_df.drop(columns=["hv_bus"],inplace=True)
            data['bus','to',node].edge_index = torch.LongTensor([edges_from, merged_df.index.tolist()])

            edges_to = merged_df["lv_bus"].tolist()
            merged_df.drop(columns=["lv_bus"],inplace=True)
            data[node,'to','bus'].edge_index = torch.LongTensor([merged_df.index.tolist(),edges_to])

        if "mv_bus" in merged_df.columns:
            edges_to2 = merged_df["mv_bus"].tolist()
            merged_df.drop(columns=["mv_bus"],inplace=True)
            data[node,'to','bus'].edge_index = torch.LongTensor([merged_df.index.tolist()+merged_df.index.tolist(),edges_to+edges_to2])

        if "bus" in merged_df.columns:
            edges_ = merged_df["bus"].tolist()
            merged_df.drop(columns=["bus"],inplace=True)
            data['bus','to',node].edge_index = torch.LongTensor([edges_, merged_df.index.tolist()])

        dataframes[node] = merged_df
        edges[node] = [edges_, edges_from, edges_to, edges_to2]
            

        print(node, len(getattr(network,node).columns), len(merged_df.columns))

    return data, edges, dataframes, scalers