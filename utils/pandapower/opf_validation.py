import torch
import pandapower as pp
import numpy as np
from itertools import chain
import ray
import copy


@ray.remote
def is_network_valid_ray(i, network, y_nodes, output_nodes, nb_gens, opf):
    return is_network_valid(i, network, y_nodes, output_nodes, nb_gens, opf)


def get_boundary_constraints_violations(i, network, y_nodes, output_nodes, nb_gens, opf, tolerance=1e-4):
    boundary_constraints = {}
    for node in y_nodes:
        values = output_nodes.get(node)
        if len(values) == 0:
            continue
        P_index = 0
        Q_index = 1
        if node == "ext_grid":
            P_index = 2
            Q_index = 3

        if node == "gen" or node == "ext_grid":
            P = values[:, P_index].reshape((-1, nb_gens[node])).cpu().numpy() * network.sn_mva
            Q = values[:, Q_index].reshape((-1, nb_gens[node])).cpu().numpy() * network.sn_mva

            P_max = getattr(network, node)["max_p_mw"].values
            P_max = np.expand_dims(P_max, 0).repeat(P.shape[0], 0)
            P_valid_max = P - P_max <= tolerance

            P_min = getattr(network, node)["min_p_mw"].values
            P_min = np.expand_dims(P_min, 0).repeat(P.shape[0], 0)
            P_valid_min = P_min - P  <= tolerance

            Q_max = getattr(network, node)["max_q_mvar"].values
            Q_max = np.expand_dims(Q_max, 0).repeat(Q.shape[0], 0)
            Q_valid_max = Q - Q_max <= tolerance

            Q_min = getattr(network, node)["min_q_mvar"].values
            Q_min = np.expand_dims(Q_min, 0).repeat(Q.shape[0], 0)
            Q_valid_min = Q_min -Q <=  tolerance

            ctr1 = np.multiply(P_valid_max.astype(np.float32),P_valid_min.astype(np.float32))
            ctr2 = np.multiply(Q_valid_max.astype(np.float32), Q_valid_min.astype(np.float32))
            boundary_constraints[node] = np.multiply(ctr1, ctr2)

        elif node == "bus":
            Vm = values[:, 4].reshape((-1, nb_gens[node])).cpu().numpy()

            Vm_max = getattr(network, node)["max_vm_pu"].values
            Vm_max = np.expand_dims(Vm_max, 0).repeat(Vm.shape[0], 0)
            valid_max = Vm -Vm_max <=  tolerance

            Vm_min = getattr(network, node)["min_vm_pu"].values
            Vm_min = np.expand_dims(Vm_min, 0).repeat(Vm.shape[0], 0)
            valid_min = Vm_min- Vm <=  tolerance

            boundary_constraints[node] = np.multiply(valid_min, valid_max)

    return boundary_constraints

def is_network_valid(i, network, y_nodes, output_nodes, nb_gens, opf, tolerance=1e-4):
    net = copy.deepcopy(network)
    valid_min_max = True
    boundaries = {}
    for node in y_nodes:
        values = output_nodes.get(node)
        if len(values) == 0:
            continue
        P_index = 0
        Q_index = 1
        if node == "ext_grid":
            P_index = 2
            Q_index = 3

        if node == "gen" or node == "ext_grid":
            P = values[:, P_index].reshape((-1, nb_gens[node])).cpu().numpy() * network.sn_mva
            Q = values[:, Q_index].reshape((-1, nb_gens[node])).cpu().numpy() * network.sn_mva

            P_max = getattr(network, node)["max_p_mw"].values
            P_max = np.expand_dims(P_max, 0).repeat(P.shape[0], 0)
            P_valid_max = P[i] <= P_max[i] if i > -1 else P - P_max <= tolerance

            P_min = getattr(network, node)["min_p_mw"].values
            P_min = np.expand_dims(P_min, 0).repeat(P.shape[0], 0)
            P_valid_min = P_min[i] <= P[i] if i > -1 else P_min - P  <= tolerance

            Q_max = getattr(network, node)["max_q_mvar"].values
            Q_max = np.expand_dims(Q_max, 0).repeat(Q.shape[0], 0)
            Q_valid_max = Q[i] <= Q_max[i] if i > -1 else Q - Q_max <= tolerance

            Q_min = getattr(network, node)["min_q_mvar"].values
            Q_min = np.expand_dims(Q_min, 0).repeat(Q.shape[0], 0)
            Q_valid_min = Q_min[i] <= Q[i] if i > -1 else Q_min -Q <=  tolerance

            valid_max = P_valid_max & Q_valid_max
            valid_min = P_valid_min & Q_valid_min

            if i > -1:
                getattr(net, "res_" + node)["q_mvar"] = Q[i]
                getattr(net, "res_" + node)["p_mw"] = P[i]

        elif node == "bus":
            Vm = values[:, 4].reshape((-1, nb_gens[node])).cpu().numpy()
            Va = values[:, 5].reshape((-1, nb_gens[node])).cpu().numpy()

            Vm_max = getattr(network, node)["max_vm_pu"].values
            Vm_max = np.expand_dims(Vm_max, 0).repeat(Vm.shape[0], 0)
            valid_max = Vm[i] <= Vm_max[i] if i > -1 else Vm <= Vm_max

            Vm_min = getattr(network, node)["min_vm_pu"].values
            Vm_min = np.expand_dims(Vm_min, 0).repeat(Vm.shape[0], 0)
            valid_min = Vm_min[i] <= Vm[i] if i > -1 else Vm_min <= Vm

            if i > -1:
                getattr(net, "res_" + node)["Vm"] = Vm[i]
                getattr(net, "res_" + node)["Va"] = Va[i]

        elif node == "line":
            continue
            # line nodes are not supported anymore in boundaries

            max_lines = getattr(network, node)[["max_i_ka"]].values + tolerance
            valid_i_from = values[i * nb_gens[node]:(i + 1) * nb_gens[node]][:, 2:3].cpu().numpy() <= max_lines
            valid_i_to = values[i * nb_gens[node]:(i + 1) * nb_gens[node]][:, 3:4].cpu().numpy() <= max_lines

            valid_min = valid_i_from
            valid_max = valid_i_to

        print(node, ": Valid min values respected:", valid_min.all(), "Valid max values respected:", valid_max.all())
        valid_min_max = valid_min_max & valid_max.all() & valid_min.all()

        boundaries = {**boundaries, node + "_min": not valid_min.all(), node + "_max": not valid_max.all()}

    run_errors = {}
    run_valid = True

    try:
        # Evaluate if there is valid powerflow when starting using the predicted values
        pp.runpp(net, init="results")
    except Exception as e:
        run_errors = pp.diagnostic(copy.deepcopy(network), report_style="compact")
        if run_errors != network.original_errors:
            run_valid = False
            print("error in opf validation", e)
            print(run_errors)

    valid = [run_valid, valid_min_max]
    return np.array(valid).astype(int), {"run": run_errors, **boundaries}


def validate_opf(networks, val_graphs, outputs, y_nodes, hetero=True, opf=1, use_ray=1):
    (out_all, val_losses_all) = outputs

    output_nodes = {node: torch.cat([e[node] for e in out_all], 0) for node in y_nodes}
    nb_gens = {node: len(networks.get("mutants")[0][node]) for node in y_nodes}

    if use_ray:
        validation = [is_network_valid_ray.remote(i, network, y_nodes, output_nodes, nb_gens, opf) for i, network in
                      enumerate(networks.get("mutants"))]
        validation_list = ray.get(validation)
    else:
        validation_list = [is_network_valid(i, network, y_nodes, output_nodes, nb_gens, opf) for i, network in
                           enumerate(networks.get("mutants"))]
    valids, errors = list(zip(*validation_list))
    # valid_networks = [validation_list[i] for i,valid in enumerate(valids) if valids]

    print("OPF validation over, nb_valid:", np.mean(valids, 0))
    return np.array(valids).astype(int), errors
