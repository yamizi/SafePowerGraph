import copy
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandapower as pp
from utils.pandapower.mutations import mutate_loads
import os
import numpy as np
import pandapower as pp

convertable_cases = ["case9","case30","case118"]
def opf(case, all_loads=None, lines_in_service=None, working_directory="./output", uniqueid="default", octave_path=None,
        batch_size=1, network=None, command="runopf"):
    # Loads are relative changes

    if octave_path is not None:
        os.environ["OCTAVE_EXECUTABLE"] = octave_path

    verbose = int(os.environ.get("VERBOSE_MATPOWER", 0))

    from oct2py import Oct2Py, Oct2PyError
    try:
        octave = Oct2Py()
    except Oct2PyError as e:
        print(e)  # noqa
        return None

    matpower_directory = f"{working_directory}/matpower"
    os.makedirs(matpower_directory, exist_ok=True)
    octave.addpath(matpower_directory)

    if network is None :
        if case in convertable_cases:
            case_method = getattr(pp.networks, case)
            network = case_method()
        else:
            octave.eval(f"mpc = loadcase('{case}');")


    pp.converter.to_mpc(network, os.path.join(matpower_directory, "base_net.mat"), init="flat")
    octave.eval(f"mpc = loadcase('base_net.mat');")
    mpc_original = octave.pull("mpc")
    buses = mpc_original.bus

    """
    PQ bus (Loads)        = 1
    PV bus (Generators)   = 2
    """
    all_buses = dict(zip(range(len(buses)), buses.tolist()))
    pq_loads = {a: v for a, v in enumerate(buses.tolist()) if (v[1] == 1 and v[2] != 0 and v[3] != 0)}

    pq = np.array(list((pq_loads.values())))
    convergence_times = []
    networks = []
    for i in range(batch_size):
        mpc = copy.deepcopy(mpc_original)
        if all_loads is not None:
            loads = all_loads[i]
            multiplier = np.ones_like(pq)
            nb_loads = min(len(loads), len(multiplier))
            multiplier[:nb_loads, 2:4] = loads[:nb_loads, 1:] + 1
        else:
            multiplier = 1
        pq_updated = pq * multiplier

        pq_loads_updated = {a: pq_updated[i].tolist() for i, a in enumerate(pq_loads.keys())}
        mpc.bus = np.array(list({**all_buses, **pq_loads_updated}.values()))
        if lines_in_service is not None:
            mpc.branch[:, 10] = np.array(lines_in_service[i], dtype=np.int8)

        octave.push("mpc", mpc)
        if verbose:
            octave.eval(f"[baseMVA, bus, gen, gencost, branch, f, success, et] = {command}(mpc);")
        else:
            octave.eval(f"evalc('[baseMVA, bus, gen, gencost, branch, f, success, et] = {command}(mpc);');")
        # octave.eval("[success, results] = runopf(mpc);")
        success = octave.pull("success")
        if not success:
            network, convergence_time = None, None
            # octave.eval(f'save("-binary", "converged_{uniqueid}.mat", "results")')
        else:
            bus_names = mpc.bus_name
            try:
                network = pp.converter.from_ppc(mpc)
            except Exception as e:
                mpc.bus_name = [e.item() for e in bus_names]
                network = pp.converter.from_ppc(mpc)
            mpc.bus = octave.pull("bus")
            mpc.gen = octave.pull("gen")
            mpc.branch = octave.pull("branch")
            convergence_time = octave.pull("et")

            pp.runpp(network)
            network.res_bus[["p_mw", "q_mvar"]] = mpc.bus[:, 2:4]
            network.res_bus[["vm_pu"]] = mpc.bus[:, 7:8]
            network.res_bus[["va_degree"]] = np.rad2deg(mpc.bus[:, 8:9])
            generators = mpc.gen[:, 0:6]
            ext_grid_bus = network.ext_grid.bus.values[0]
            ext_grid_index = generators[:, 0].tolist().index(ext_grid_bus)
            network.res_ext_grid[["p_mw", "q_mvar"]] = generators[ext_grid_index, 1:3]
            network.res_gen[["p_mw", "q_mvar"]] = np.delete(generators, ext_grid_index, axis=0)[:, 1:3]
            # network.res_gen[['vm_pu']] = np.delete(generators,ext_grid_index, axis=0)[:,5]

        networks.append(network)
        convergence_times.append(convergence_time)

    octave.exit()

    return networks, convergence_times


if __name__ == "__main__":
    case = "case1354pegase"
    # case="case9"
    case_method = getattr(pp.networks, case)
    net = case_method()

    mutation_rate = 0.7
    octave_path = "C:/PortableApps/GNUOctavePortable/App/Octave64/mingw64/bin/octave-cli.exe"
    network, masked_loads = mutate_loads(net, mutation_rate=mutation_rate, relative=True)

    opf(octave_path=octave_path, case=case, loads=masked_loads)
