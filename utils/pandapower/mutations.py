import pandapower as pp
import numpy as np


def build_costs(net, costs):
    for cost in costs:
        et, index, prices = cost

        for i, p in prices.items():
            if p is None:
                prices[i] = 0

        found = net.poly_cost[(net.poly_cost.element == index) & (net.poly_cost.et == et)]
        if len(net.poly_cost) and len(found) > 0:
            print("updating cost of ", et, index, "to", prices)
            net.poly_cost.loc[found.index, list(prices.keys())] = list(prices.values())
        else:
            print("setting new cost of ", et, index, "to", prices)
            pp.create_poly_cost(net, index, et, check=False, **prices)


def mutate_loads(network, min_p=0, max_p=0, min_q=0, max_q=0, clip=False, mutation_rate=0.7, relative=False,
                 factor=2, reactive_weight=0.2, reuse_loads=None):
    # min_p=-0.08, max_p=0.1,min_q=-0.08, max_q=0.1

    if len(network.load) == 0:
        return network


    if not relative:
        if min_p == max_p == min_q == max_q == 0:
            min_p = network.load.p_mw.min() / factor
            max_p = network.load.p_mw.max() / factor
            min_q = network.load.q_mvar.min() / factor
            max_q = network.load.q_mvar.max() / factor

            if clip:
                min_p = max(min_p, network.load.p_mw.min() / factor)
                max_p = min(max_p, network.load.p_mw.max() / factor)

                min_q = max(min_q, network.load.q_mvar.min() / factor)
                max_q = min(max_q, network.load.q_mvar.max() / factor)

                print("clipping min and max loads to {} and {}".format(min_p, max_p))
    else:
        min_p = -0.1  if min_p==0 else min_p
        max_p = 0.1 if max_p==0 else max_p
        min_q = -0.1  if min_q==0 else min_q # -0.08
        max_q = 0.1 if max_q==0 else max_q

    if reuse_loads is None:
        loads = [[i, np.random.uniform(min_p, max_p) * factor, np.random.uniform(min_q, max_q) * factor] for i in
                 range(len(network.load))]

        mask = np.random.choice(len(loads), (int(len(loads) * mutation_rate)), replace=False)
        masked_loads = np.array(loads)[mask]
        loads = np.array(loads)[mask]
        # print("updating loads", masked_loads)
        if relative:
            masked_loads[:, 1] = (masked_loads[:, 1] + 1) * network.load.loc[masked_loads[:, 0].astype(int), "p_mw"]
            masked_loads[:, 2] = (masked_loads[:, 2] * reactive_weight + 1) * network.load.loc[
                masked_loads[:, 0].astype(int), "q_mvar"]
    else:
        masked_loads = reuse_loads
    network.load.loc[masked_loads[:, 0].astype(int), "p_mw"] = masked_loads[:, 1]
    network.load.loc[masked_loads[:, 0].astype(int), "q_mvar"] = masked_loads[:, 2]

    return network, loads


def mutate_costs(network, min_cost=1, max_cost=10, clip=False, mutation_rate=0.7):
    if clip and len(network.poly_cost):
        min_cost = min(max(min_cost, network.poly_cost.cp1_eur_per_mw.min()), network.poly_cost.cp1_eur_per_mw.max())
        max_cost = min(max(max_cost, network.poly_cost.cp1_eur_per_mw.min()), network.poly_cost.cp1_eur_per_mw.max())

        print("clipping min and max costs to {} and {}".format(min_cost, max_cost))
    costs_grids = [("ext_grid", i, {"cp1_eur_per_mw": np.random.randint(min_cost, max_cost)}) for i in
                   range(len(network.ext_grid))]
    costs_gen = [("gen", i, {"cp1_eur_per_mw": np.random.randint(min_cost, max_cost)}) for i in range(len(network.gen))]
    costs_sgen = [("sgen", i, {"cp1_eur_per_mw": np.random.randint(min_cost, max_cost)}) for i in
                  range(len(network.sgen))]

    costs = costs_grids + costs_gen + costs_sgen
    mask = np.random.choice(len(costs), (int(len(costs) * mutation_rate)), replace=False)
    masked_costs = list(np.array(costs)[mask])
    build_costs(network, masked_costs)

    return network, masked_costs


def disable_lines(network, mutation_rate=0.7, num_lines_disable=1):
    nminus1_cases = network.line.index.values
    remaining_trials = 1
    while remaining_trials > 0:
        remaining_trials -= 1
        disable_prob = np.random.random(1)
        lines_to_disable = np.random.choice(nminus1_cases, num_lines_disable,
                                            False) if disable_prob < mutation_rate else []
        # try:
        for i in lines_to_disable:
            network["line"].at[i, 'in_service'] = False

        return network, network["line"]['in_service']
