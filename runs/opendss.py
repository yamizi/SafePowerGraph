import os.path
import sys
import copy
import warnings
import pandapower as pp
from utils.pandapower.mutations import mutate_loads
import os
import numpy as np
import opendssdirect as dss
import time

warnings.simplefilter(action='ignore', category=FutureWarning)

convertable_cases = ["case9", "case30", "case118"]


def pandapower_to_opendss(network, dss_case_path):
    try:
        with open(dss_case_path, "w") as f:
            f.write("Clear\n")
            f.write("New Circuit.MyCircuit BasekV=345.0\n")

            # Convert lines
            for index, line in network.line.iterrows():
                from_bus = f"bus{line.from_bus}" #network.bus.loc[line.from_bus, 'name']
                to_bus = f"bus{line.to_bus}" #network.bus.loc[line.to_bus, 'name']
                r = line.r_ohm_per_km * line.length_km
                x = line.x_ohm_per_km * line.length_km
                f.write(
                    f"New Line.Line{index} Bus1={from_bus} Bus2={to_bus} R1={r} X1={x} Length={line.length_km} Units=km\n")

            # Convert loads
            for index, load in network.load.iterrows():
                bus = f"bus{load.bus}" #network.bus.loc[load.bus, 'name']
                p_mw = load.p_mw
                q_mvar = load.q_mvar
                f.write(
                    f"New Load.Load{index} Bus1={bus} Phases=3 Conn=Wye Model=1 kV={network.bus.loc[load.bus, 'vn_kv']} kW={p_mw * 1000} kvar={q_mvar * 1000}\n")

            # Convert generators
            for index, gen in network.gen.iterrows():
                bus = f"bus{gen.bus}" #network.bus.loc[gen.bus, 'name']
                p_mw = gen.p_mw
                vm_pu = gen.vm_pu
                f.write(
                    f"New Generator.Gen{index} Bus1={bus} Phases=3 kV={network.bus.loc[gen.bus, 'vn_kv']} kW={p_mw * 1000} Model=1\n")

            # Convert synchronous condensers
            for index, sync_condenser in network.sgen.iterrows():
                bus = f"bus{sync_condenser.bus}" #network.bus.loc[sync_condenser.bus, 'name']
                p_mw = sync_condenser.p_mw
                vm_pu = sync_condenser.vm_pu
                f.write(
                    f"New SynchronousCond.SCond{index} Bus1={bus} Phases=3 kV={network.bus.loc[sync_condenser.bus, 'vn_kv']} kW={p_mw * 1000} Model=1\n")

            # Convert transformers
            for index, trafo in network.trafo.iterrows():
                hv_bus = f"bus{trafo.hv_bus}" #network.bus.loc[trafo.hv_bus, 'name']
                lv_bus = f"bus{trafo.lv_bus}" #network.bus.loc[trafo.lv_bus, 'name']
                sn_mva = trafo.sn_mva
                vn_hv_kv = trafo.vn_hv_kv
                vn_lv_kv = trafo.vn_lv_kv
                vkr_percent = trafo.vkr_percent
                vk_percent = trafo.vk_percent
                pfe_kw = trafo.pfe_kw
                i0_percent = trafo.i0_percent
                shift_degree = trafo.shift_degree
                f.write(
                    f"New Transformer.Xfmr{index} Buses=({hv_bus}, {lv_bus}) Phases=3 kvs=({vn_hv_kv}, {vn_lv_kv}) kvas=({sn_mva * 1000}, {sn_mva * 1000}) Xhl={vk_percent} R={vkr_percent}\n")

            # Convert external grid
            for index, ext_grid in network.ext_grid.iterrows():
                bus = f"bus{ext_grid.bus}" #network.bus.loc[ext_grid.bus, 'name']
                f.write(
                    f"New Vsource.Source{index} Bus1={bus} Phases=3 BasekV={network.bus.loc[ext_grid.bus, 'vn_kv']}\n")

            f.write("Solve\n")

        print(f"DSS case file saved to: {dss_case_path}")
    except Exception as e:
        print(f"Error in pandapower_to_opendss: {e}")
    return dss_case_path


def run_opendss(dss_case_path):
    try:
        dss.run_command(f"Redirect {dss_case_path}")
        start_time = time.time()
        dss.run_command("Solve")
        convergence_time = time.time() - start_time

        success = dss.Solution.Converged()
        if not success:
            return None, None, None, None, None, convergence_time

        # Extract results
        buses = dss.Circuit.AllBusNames()
        all_bus_voltages = dss.Circuit.AllBusVolts()
        bus_voltages = {bus: (all_bus_voltages[i*2], all_bus_voltages[i*2+1]) for i, bus in enumerate(buses)}

        # Extract other necessary results
        generators = dss.Generators.AllNames()
        gen_results = {}
        for gen in generators:
            dss.Generators.Name(gen)
            gen_results[gen] = dss.CktElement.Powers()

        loads = dss.Loads.AllNames()
        load_results = {}
        for load in loads:
            dss.Loads.Name(load)
            load_results[load] = dss.CktElement.Powers()

        # Extract synchronous condenser results
        sync_conds = dss.SwtControls.AllNames()
        sync_cond_results = {}
        for sc in sync_conds:
            dss.SwtControls.Name(sc)
            sync_cond_results[sc] = dss.CktElement.Powers()

        #Extract slack results
        dss.Circuit.SetActiveElement('Vsource.source')
        slack_results = dss.CktElement.Powers()

        return bus_voltages, gen_results, slack_results, load_results, sync_cond_results, convergence_time
    except Exception as e:
        print(f"Error in run_opendss: {e}")
        return None, None, None, None, None, None


def update_pandapower_results(network, bus_voltages, gen_results, slack_results, load_results, sync_cond_results, trafo_results):
    try:
        # Update bus results
        for bus, (real, imag) in bus_voltages.items():
            bus_index = int(bus.lower().replace("bus",""))
            vm_pu = np.sqrt(real**2 + imag**2) / network.bus.vn_kv[bus_index]
            va_degree = np.arctan2(imag, real) * 180 / np.pi
            network.res_bus.loc[bus_index, "vm_pu"] = vm_pu
            network.res_bus.loc[bus_index, "va_degree"] = va_degree

        # Update generator results
        for gen, powers in gen_results.items():
            gen_index = int(gen.lower().replace("gen",""))
            network.res_gen.loc[gen_index, "p_mw"] = powers[0] / 1000
            network.res_gen.loc[gen_index, "q_mvar"] = powers[1] / 1000

        network.res_ext_grid.loc[0, "p_mw"] = slack_results[0] / 1000  # Convert kW to MW
        network.res_ext_grid.loc[0, "q_mvar"] = slack_results[1] / 1000  # Convert kVAR to MVAR

        # Update load results
        for load, powers in load_results.items():
            load_index = int(load.lower().replace("load",""))
            network.res_load.loc[load_index, "p_mw"] = -powers[0] / 1000
            network.res_load.loc[load_index, "q_mvar"] = -powers[1] / 1000

        # Update synchronous condenser results
        for sc, powers in sync_cond_results.items():
            sc_index = int(sc.lower().replace("scond",""))
            network.res_sgen.loc[sc_index, "p_mw"] = powers[0] / 1000
            network.res_sgen.loc[sc_index, "q_mvar"] = powers[1] / 1000

        # Update transformer results
        for trafo, powers in trafo_results.items():
            trafo_index = int(trafo.lower().replace("xfmr",""))
            network.res_trafo.loc[trafo_index, "p_hv_mw"] = powers[0] / 1000
            network.res_trafo.loc[trafo_index, "q_hv_mvar"] = powers[1] / 1000
            network.res_trafo.loc[trafo_index, "p_lv_mw"] = powers[2] / 1000
            network.res_trafo.loc[trafo_index, "q_lv_mvar"] = powers[3] / 1000

    except Exception as e:
        print(f"Error in update_pandapower_results: {e}")


def pf(case, all_loads=None, lines_in_service=None, working_directory="./output", uniqueid="default", batch_size=1,
       network=None):
    convergence_times = []
    networks = []

    for i in range(batch_size):

        if network is None:
            # Load pandapower network
            case_method = getattr(pp.networks, case)
            network = case_method()

            # Mutate loads
            if all_loads is not None:
                network, masked_loads = mutate_loads(network, mutation_rate=0.7, relative=True, reuse_loads=all_loads)

        # Create OpenDSS case file
        dss_output_directory = os.path.join(working_directory, "opendss")
        os.makedirs(dss_output_directory, exist_ok=True)
        dss_case_path = pandapower_to_opendss(network, os.path.join(dss_output_directory, f"{case}_{uniqueid}.dss"))

        # Run power flow in OpenDSS
        bus_voltages, gen_results, load_results, slack_results, sync_cond_results, convergence_time = run_opendss(dss_case_path)

        if bus_voltages is None:
            networks.append(None)
            convergence_times.append(convergence_time)
            continue

        # Run initial power flow to populate result tables
        if not hasattr(network,"res_bus") or len(network.res_bus)==0:
            pp.runpp(network)

        original_network = copy.deepcopy(network)
        # Update pandapower network with results
        update_pandapower_results(network, bus_voltages, gen_results, slack_results, load_results, trafo_results)

        networks.append(network)
        convergence_times.append(convergence_time)

    return networks, convergence_times


if __name__ == "__main__":
    case = "case9"  # Example case
    mutation_rate = 0.7
    case_method = getattr(pp.networks, case)
    network = case_method()

    # Apply load mutation
    network, masked_loads = mutate_loads(network, mutation_rate=mutation_rate, relative=True)

    # Execute OPF with OpenDSS
    networks, convergence_times = pf(case=case, all_loads=masked_loads)
    for i, network in enumerate(networks):
        if network is not None:
            print(f"Results for batch {i + 1}:")
            print(network.res_bus)
            print(f"Convergence time: {convergence_times[i]}")
        else:
            print(f"Batch {i + 1} failed to converge.")
