import torch
import numpy as np
from torch_geometric.data import HeteroData


def calculate_boundary_violations(data: HeteroData, use_solution=False, use_prediction=None):
    violations = {
        'generator_p': [],
        'generator_q': [],
        'bus_vm': []
    }

    for i in range(len(data['bus'].ptr) - 1):
        bus_slice = slice(data['bus'].ptr[i], data['bus'].ptr[i + 1])
        gen_slice = slice(data['generator'].ptr[i], data['generator'].ptr[i + 1])

        # Generator and bus values
        if use_prediction is not None:
            gen_p = use_prediction['generator'][gen_slice, 0] / data['generator'].x[gen_slice, 0]
            gen_q = use_prediction['generator'][gen_slice, 1]
            vm = use_prediction['bus'][bus_slice, 1] / data['bus'].x[bus_slice, 0]
        else:
            if use_solution:
                gen_p = data['generator'].y[gen_slice, 0] / data['generator'].x[gen_slice, 0]
                gen_q = data['generator'].y[gen_slice, 1]
                vm = data['bus'].y[bus_slice, 1] / data['bus'].x[bus_slice, 0]
            else:
                gen_p = data['generator'].x[gen_slice, 1] / data['generator'].x[gen_slice, 0]
                gen_q = data['generator'].x[gen_slice, 4] / data['generator'].x[gen_slice, 0]
                vm = data['bus'].x[bus_slice, 0] / data['bus'].x[bus_slice, 0]

        # Constraint parameters (always from initial data)
        p_min = data['generator'].x[gen_slice, 2] / data['generator'].x[gen_slice, 0]
        p_max = data['generator'].x[gen_slice, 3] / data['generator'].x[gen_slice, 0]
        q_min = data['generator'].x[gen_slice, 5] / data['generator'].x[gen_slice, 0]
        q_max = data['generator'].x[gen_slice, 6] / data['generator'].x[gen_slice, 0]
        v_min = data['bus'].x[bus_slice, 2] / data['bus'].x[bus_slice, 0]
        v_max = data['bus'].x[bus_slice, 3] / data['bus'].x[bus_slice, 0]

        # Violation calculations
        violations['generator_p'].append(
            torch.max(gen_p - p_max, p_min - gen_p).clamp(min=0)
        )
        violations['generator_q'].append(
            torch.max(gen_q - q_max, q_min - gen_q).clamp(min=0)
        )
        violations['bus_vm'].append(
            torch.max(vm - v_max, v_min - vm).clamp(min=0)
        )

    return violations

def calculate_mismatch_batch(batch: HeteroData, use_solution=False, use_prediction=None):
    # Get batch separation pointers
    bus_ptr = batch['bus'].ptr


    mismatches = []

    # Process each graph in the batch separately
    for idx in range(len(bus_ptr) - 1):

        # Extract components for current graph
        current_data = HeteroData()
        current_predictions = {} if use_prediction is not None else None
        for node_type in batch.node_types:
            start = batch[node_type].ptr[idx]
            end = batch[node_type].ptr[idx + 1]
            current_data[node_type].x = batch[node_type].x[start:end]
            if node_type == 'bus' or node_type == 'generator':
                current_data[node_type].y = batch[node_type].y[start:end]
                if use_prediction is not None:
                    current_predictions[node_type] = use_prediction[node_type][start:end]

        for edge_type in batch.edge_types:
            src_type, _, dst_type = edge_type
            src_ptr = batch[src_type].ptr
            dst_ptr = batch[dst_type].ptr

            # Get slice indices for current graph
            src_start = src_ptr[idx]
            src_end = src_ptr[idx + 1]
            dst_start = dst_ptr[idx]
            dst_end = dst_ptr[idx + 1]

            # Filter edges and adjust indices
            mask = (batch[edge_type].edge_index[0] >= src_start) & \
                   (batch[edge_type].edge_index[0] < src_end) & \
                   (batch[edge_type].edge_index[1] >= dst_start) & \
                   (batch[edge_type].edge_index[1] < dst_end)

            filtered_edges = batch[edge_type].edge_index[:, mask] \
                             - torch.tensor([[src_start], [dst_start]], device=batch[edge_type].edge_index.device)

            current_data[edge_type].edge_index = filtered_edges

            # Extract edge attributes if present
            if hasattr(batch[edge_type], 'edge_attr'):
                current_data[edge_type].edge_attr = batch[edge_type].edge_attr[mask]

        # Calculate mismatch for individual graph
        deltaP, deltaQ = calculate_single_graph_mismatch(current_data, use_solution, current_predictions)
        mismatches.append((deltaP, deltaQ))

    return mismatches


def calculate_single_graph_mismatch(graph: HeteroData,use_solution=False, use_prediction=None):
    # Build Ybus matrix using edge attributes
    n_buses = graph['bus'].get('x').size(0)
    device = graph['bus'].get('x').device

    # Voltage and generation initialization
    if use_prediction is not None:
        V_ang = use_prediction['bus'][:, 0]  # Voltage angle from prediction
        V_mag = use_prediction['bus'][:, 1] # Voltage magnitude from prediction

        gen_p = use_prediction['generator'][:, 0]  # Optimal Pg
        gen_q = use_prediction['generator'][:, 1]  # Optimal Qg
    else:
        if use_solution:
            V_ang = graph['bus'].get('y')[:, 0]  # Voltage angle from solution
            V_mag = graph['bus'].get('y')[:, 1]  # Voltage magnitude from solution

            gen_p = graph['generator'].get('y')[:, 0]  # Optimal Pg
            gen_q = graph['generator'].get('y')[:, 1]  # Optimal Qg
        else:
            V_mag = graph['bus'].get('x')[:, 0]  # Initial voltage magnitude
            V_ang = 0  # Initial voltage angle = 0

            gen_p = graph['generator'].get('x')[:, 1]  # Initial Pg
            gen_q = graph['generator'].get('x')[:, 4]  # Initial Qg

    m_base = graph['generator'].x[:, 0]
    V_mag = V_mag / graph['bus'].x[:, 0]
    gen_p = gen_p / m_base
    gen_q = gen_q / m_base

    V_ang = torch.tensor([np.pi / 4], dtype=torch.float32)  # Example angle

    # Convert to complex tensor using 1j
    complex_phase = 1j * V_ang  # This creates complex dtype automatically

    # Now compute exponential
    V = V_mag * torch.exp(complex_phase).to(V_mag.device)

    # Initialize Ybus with shunt admittances
    Ybus = torch.zeros(n_buses, n_buses, dtype=torch.complex64, device=device)

    # Add AC line contributions
    if ('bus', 'ac_line', 'bus') in graph.edge_types:
        ac_line_edges = graph[('bus', 'ac_line', 'bus')]
        for idx in range(ac_line_edges.edge_index.size(1)):
            i, j = ac_line_edges.edge_index[:, idx]
            br_r = ac_line_edges.edge_attr[idx, 4]
            br_x = ac_line_edges.edge_attr[idx, 5]
            b_fr = ac_line_edges.edge_attr[idx, 2]
            b_to = ac_line_edges.edge_attr[idx, 3]

            Z = torch.complex(br_r, br_x).to(device)
            Y = 1 / Z
            Ybus[i, i] += Y + torch.complex(torch.tensor(0.0, device=b_fr.device), b_fr / 2).to(device)
            Ybus[j, j] += Y + torch.complex(torch.tensor(0.0, device=b_fr.device), b_to / 2).to(device)
            Ybus[i, j] -= Y
            Ybus[j, i] -= Y

    # Add transformer contributions
    if ('bus', 'transformer', 'bus') in graph.edge_types:
        tf_edges = graph[('bus', 'transformer', 'bus')]
        for idx in range(tf_edges.edge_index.size(1)):
            i, j = tf_edges.edge_index[:, idx]
            br_r = tf_edges.edge_attr[idx, 2]
            br_x = tf_edges.edge_attr[idx, 3]
            tap = tf_edges.edge_attr[idx, 7]
            b_fr = tf_edges.edge_attr[idx, 9]
            b_to = tf_edges.edge_attr[idx, 10]

            Z = torch.complex(br_r, br_x)
            Y = 1 / (Z * tap ** 2)
            Ybus[i, i] += Y + torch.complex(torch.tensor(0.0, device=b_fr.device), b_fr / 2).to(device)
            Ybus[j, j] += Y + torch.complex(torch.tensor(0.0, device=b_fr.device), b_to / 2).to(device)
            Ybus[i, j] -= Y / tap
            Ybus[j, i] -= Y / tap

    # Add shunt contributions
    if 'shunt' in graph.node_types:
        for shunt_idx in range(graph['shunt'].x.size(0)):
            bus_idx = graph[('shunt', 'shunt_link', 'bus')].edge_index[1, shunt_idx]
            gs = graph['shunt'].x[shunt_idx, 0]
            bs = graph['shunt'].x[shunt_idx, 1]
            Ybus[bus_idx, bus_idx] += torch.complex(gs, bs)

    # Calculate power injections
    I = torch.mv(Ybus, V.conj())
    S = V * I.conj()
    P_calc = S.real
    Q_calc = S.imag

    # Aggregate generations and loads
    P_inj = torch.zeros(n_buses, device=device)
    Q_inj = torch.zeros(n_buses, device=device)

    # Add generator injections
    gen_buses = graph[('generator', 'generator_link', 'bus')].edge_index[1]
    P_inj[gen_buses] += gen_p
    Q_inj[gen_buses] += gen_q

    # Subtract loads
    load_buses = graph[('load', 'load_link', 'bus')].edge_index[1]
    P_inj[load_buses] -= graph['load'].x[:, 0]  # Assuming PD is first feature
    Q_inj[load_buses] -= graph['load'].x[:, 1]  # Assuming QD is second feature

    # Calculate mismatches
    deltaP = P_inj - P_calc
    deltaQ = Q_inj - Q_calc

    # Apply bus type masks
    bus_types = graph['bus'].x[:, 1].long()  # Assuming bus type is second feature
    pq_mask = (bus_types == 1)
    pv_mask = (bus_types == 2).int() +  (bus_types == 3).int()

    return deltaP[pv_mask | pq_mask], deltaQ[pq_mask]