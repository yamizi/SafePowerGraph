import torch
import numpy as np
from utils.pandapower.normalization import MAX_ANGLE
import time
from torch.nn.functional import mse_loss
import os


def relative_loss(yhat, y):
    criterion = torch.nn.L1Loss(reduction="none")
    # criterion = torch.nn.MSELoss(reduction="none")
    return criterion(yhat, y) / yhat.abs()


def node_loss(yhat, y, mask=None):
    criterions = [torch.nn.MSELoss(reduction="none"), torch.nn.L1Loss(reduction="none")]
    if mask is None:
        losses = [criterion(yhat, y[:, :yhat.shape[1]]).unsqueeze(0) for criterion in criterions]
    else:
        losses = [criterion(yhat.flatten(), y.flatten()[mask.bool().flatten()]).unsqueeze(0) for criterion in
                  criterions]

    return torch.cat(losses).sum(0)


def boundary_loss(boundaries, y, node=""):
    # minp = boundaries[:,0] # maxp = boundaries[:,1]
    # minq = boundaries[:,2] # maxq = boundaries[:,3]
    # minVm = boundaries[:,4] # maxVm = boundaries[:,5]
    # minVa = boundaries[:,6] # maxVa = boundaries[:,7]
    # mini_from_ka = boundaries[:,12] # maxi_from_ka = boundaries[:,13]
    # mini_to_ka = boundaries[:,14] # maxi_to_ka = boundaries[:,15]

    boundary_losses = [torch.max(torch.zeros_like(boundaries[:, 2 * i]), boundaries[:, 2 * i] - y[:, i]) + torch.max(
        torch.zeros_like(boundaries[:, 2 * i + 1]), y[:, i] - boundaries[:, 2 * i + 1]) for i in range(y.shape[1])
                       if torch.isnan(boundaries[:, 2 * i]).sum() == 0]
    return torch.stack(boundary_losses).sum(0)
    # return torch.max(torch.zeros_like(minp),minp-y[:,0]) + torch.max(torch.zeros_like(maxp),y[:,0]-maxp) + torch.max(torch.zeros_like(minq),minq-y[:,1]) + torch.max(torch.zeros_like(maxq),y[:,1]-maxq)


def get_loss_parameters(neighboorhood, data, out):
    ## Caching repetitive variables that remain the same in all batches (except last one when incomplete)
    if neighboorhood is None or len(neighboorhood[2]) != len(out.get("gen")):

        bus_to_line_list = list(zip(*data.edge_index_dict.get(('bus', 'to', 'line')).cpu().tolist()))
        line_to_bus_list = list(zip(*data.edge_index_dict.get(('line', 'to', 'bus')).cpu().tolist()))

        bus_to_gen_index, gen_index = data.edge_index_dict.get(('bus', 'to', 'gen')).cpu().tolist()
        bus_to_ext_index, ext_index = data.edge_index_dict.get(('bus', 'to', 'ext_grid')).cpu().tolist()

        bus_to_bus_dict = [
            (start, end, l1) for
            (start, l1) in bus_to_line_list for (l2, end) in line_to_bus_list if (l1 == l2 and start != end)]

        i, j = [a[0] for a in bus_to_bus_dict], [a[1] for a in bus_to_bus_dict]

        bus_size = len(out.get("bus"))
        buses = torch.LongTensor(i)
        buses_t = torch.LongTensor(j)
        mask = torch.stack([buses == a for a in np.arange(bus_size)])

        mask_t = torch.stack([buses_t == a for a in np.arange(bus_size)])

        neighboorhood = bus_to_line_list, line_to_bus_list, bus_to_gen_index, gen_index, bus_to_ext_index, ext_index, bus_to_bus_dict, i, j, mask, mask_t

    else:
        bus_to_line_list, line_to_bus_list, bus_to_gen_index, gen_index, bus_to_ext_index, ext_index, bus_to_bus_dict, i, j, mask, mask_t = neighboorhood

    # line features are: 'std_type', 'length_km', 'r_ohm_per_km', 'x_ohm_per_km', 'c_nf_per_km','g_us_per_km'
    line_features = data.x_dict.get("line")[:, :6]
    # bus features are : 'vn_kv', 'in_service', 'min_vm_pu', 'max_vm_pu', 'p_mw', 'q_mvar'
    bus_features = data.x_dict.get("bus")[:, -2:]
    V_base_kv = data['bus'].x[i, 0]
    S_base = data.sn_mva[0] if isinstance(data.sn_mva, torch.Tensor) and len(data.sn_mva)>1 else data.sn_mva
    #S_base = 1 # the active and reactive powers are already normalized
    Z_base = V_base_kv ** 2 / S_base
    f_hz = data.f_hz[0] if isinstance(data.f_hz, torch.Tensor) and len(data.f_hz)>1  else data.f_hz

    # a[0]: index of start bus
    # a[1]: index of end bus
    # a[2]: index of line connecting them

    # (i, j, r_ij, x_ij)
    r, x, b = torch.stack(
        [line_features[a[2], 2] * line_features[a[2], 3] / Z_base[0] for a in bus_to_bus_dict]), torch.stack(
        [line_features[a[2], 2] * line_features[a[2], 4] / Z_base[0] for a in bus_to_bus_dict]), torch.stack(
        [line_features[a[2], 2] * line_features[a[2], 5] * 2 * np.pi * f_hz * Z_base[0] / 2 / 10 ** 9 for a in
         bus_to_bus_dict])  #

    return neighboorhood , S_base, mask, mask_t, bus_features, bus_to_line_list, line_to_bus_list, bus_to_gen_index, gen_index, bus_to_ext_index, ext_index, bus_to_bus_dict, i, j, mask, mask_t, r.squeeze(), x.squeeze(), b.squeeze()


def get_power_imbalance(r,x,b,vm_i,vm_j,va_i,va_j,mask, mask_t,bus_features,bus_to_gen_index,sn_mva,gen_buses,bus_to_ext_index,ext_buses ):
    ## First side of the loss

    r_x = torch.stack([r, x], dim=1)
    zm_ij = torch.norm(r_x, p=2, dim=-1, keepdim=True).squeeze(
        1)  # (num_edges, 1) NOTE (r**2+x**2)**0.5 should be non-zero
    za_ij = torch.acos(r / zm_ij)  # (num_edges, 1)
    ym_ij = 1 / (zm_ij + 1e-6)  # (num_edges, 1)
    ya_ij = -za_ij  # (num_edges, 1)
    g_ij = ym_ij * torch.cos(ya_ij)  # (num_edges, 1)
    b_ij = ym_ij * torch.sin(ya_ij)  # (num_edges, 1)
    g_ij2 = r / (r ** 2 + x ** 2)
    b_ij2 = -x / (r ** 2 + x ** 2)

    Pij = g_ij * vm_i ** 2 - vm_i * vm_j * (g_ij * torch.cos(va_i - va_j) + b_ij * torch.sin(va_i - va_j))
    Qij = -b_ij * vm_i ** 2 - vm_i * vm_j * (
            g_ij * torch.sin(va_i - va_j) - b_ij * torch.cos(va_i - va_j)) - b * vm_i ** 2
    Pji = g_ij * vm_j ** 2 - vm_i * vm_j * (g_ij * torch.cos(va_j - va_i) + b_ij * torch.sin(va_j - va_i))
    Qji = -b_ij * vm_j ** 2 - vm_i * vm_j * (
            g_ij * torch.sin(va_j - va_i) - b_ij * torch.cos(va_j - va_i)) - b * vm_j ** 2

    mask = mask.float()
    mask = mask.to(Pij.device)  # or mask.cuda()
    mask_t = mask_t.float()
    mask_t = mask_t.to(Pji.device)
    Pi = torch.matmul(mask, Pij) + torch.matmul(mask_t, Pji)
    Qi = torch.matmul(mask, Qij) + torch.matmul(mask_t, Qji)

    ## Second side of the loss

    bus_generator = torch.zeros_like(bus_features)
    bus_ext = torch.zeros_like(bus_features)
    bus_generator[bus_to_gen_index, :] = gen_buses
    bus_ext[bus_to_ext_index, :] = ext_buses

    bus_true = -bus_features/sn_mva + bus_generator + bus_ext
    Pi_true = bus_true[:, 0]
    Qi_true = bus_true[:, 1]

    loss = [mse_loss(Pi, Pi_true).unsqueeze(0), mse_loss(Qi, Qi_true).unsqueeze(0)]

    return loss
def mixed_power_imbalance_loss(data,out,neighboorhood=None):
    begin = time.time()

    neighboorhood, S_base, mask, mask_t, bus_features, bus_to_line_list, line_to_bus_list, bus_to_gen_index, gen_index, bus_to_ext_index, ext_index, bus_to_bus_dict, i, j, mask, mask_t, r, x, b = get_loss_parameters(
        neighboorhood, data, out)

    vm_i_out, vm_j_out, va_i_out, va_j_out, gen_buses_out, ext_buses_out = get_variables(False, data, out, i, j, gen_index, ext_index)
    vm_i_true, vm_j_true, va_i_true, va_j_true, gen_buses_true, ext_buses_true = get_variables(True, data, out, i, j,
                                                                                         gen_index, ext_index)

    # Loss 1  left: true, right: pred

    loss1 = get_power_imbalance(r,x,b,vm_i_true,vm_j_true,va_i_true,va_j_true,mask, mask_t,bus_features,bus_to_gen_index,S_base,gen_buses_out,bus_to_ext_index,ext_buses_out)
    loss2 = get_power_imbalance(r, x, b, vm_i_out, vm_j_out, va_i_out, va_j_out, mask, mask_t, bus_features, bus_to_gen_index,
                                S_base, gen_buses_true, bus_to_ext_index, ext_buses_true)

    duration = time.time() - begin
    loss = loss1+loss2
    return torch.cat(loss, dim=-1), neighboorhood, duration, S_base  # (num_edges, 2)

def get_variables(ground_truth, data, out,i,j, gen_index, ext_index):
    sn_mva = data.sn_mva[0] if isinstance(data.sn_mva, torch.Tensor) and len(data.sn_mva)>1 else data.sn_mva
    angle=data.angle[0] if isinstance(data.angle, torch.Tensor) else data.angle

    if ground_truth:
        vm_i = data["bus"].y[i, 0]
        vm_j = data["bus"].y[j, 0]

        va_i = torch.deg2rad(data["bus"].y[i, 1]) if angle=="deg" else data["bus"].y[i, 1]
        va_j = torch.deg2rad(data["bus"].y[j, 1]) if angle=="deg" else data["bus"].y[j, 1]

        gen_buses = data["gen"].y[gen_index, :]
        ext_buses = data["ext_grid"].y[ext_index, :]
    else:
        vm_i = out.get("bus")[i, 4]
        vm_j = out.get("bus")[j, 4]

        va_i = torch.deg2rad(out.get("bus")[i, 5]) if angle=="deg" else out.get("bus")[i, 5]
        va_j = torch.deg2rad(out.get("bus")[j, 5]) if angle=="deg" else out.get("bus")[j, 5]

        gen_buses = out.get("gen")[gen_index, 0:2]
        ext_buses = out.get("ext_grid")[ext_index, 2:4]

    return vm_i, vm_j, va_i, va_j, gen_buses, ext_buses


def power_imbalance_loss(data, out, neighboorhood=None, ground_truth=False, version="2"):
    """calculate injected power Pji

    Formula:
    $$
    P_{ji} = V_m^i*V_m^j*Y_{ij}*\cos(V_a^i-V_a^j-\theta_{ij})
            -(V_m^i)^2*Y_{ij}*\cos(-\theta_{ij})
    $$
    $$
    Q_{ji} = V_m^i*V_m^j*Y_{ij}*\sin(V_a^i-V_a^j-\theta_{ij})
            -(V_m^i)^2*Y_{ij}*\sin(-\theta_{ij})
    $$

    Input:
    data: the GNN features and labels
    output: the ouput of the GCN, P and Q for genertors and external grid and Vm,Va for loads
    ground_truth: True to compute the losses on the ground truth, should return close to 0 (for debugging purpose)

    Return:
        MSE loss on Pi and Qi between the two terms of the loss

    """
    print("running power imabalance loss")
    details = {}
    if version=="3":
        cat_loss, neighboorhood, duration, S_base =  mixed_power_imbalance_loss(data,out,neighboorhood)
    else:
        begin = time.time()
        ground_truth = int(os.environ.get("USE_GT_POWERIMBALANCE", ground_truth))

        neighboorhood, S_base, mask, mask_t, bus_features, bus_to_line_list, line_to_bus_list, bus_to_gen_index, gen_index, bus_to_ext_index, ext_index, bus_to_bus_dict, i, j, mask, mask_t, r, x, b = get_loss_parameters(neighboorhood, data, out)

        vm_i, vm_j, va_i, va_j, gen_buses, ext_buses = get_variables(ground_truth, data, out,i,j, gen_index, ext_index)


        ## First side of the loss

        ### V1 (from Powerflownet not correct anymore)
        if version=="1":
            g_ij = r / (r ** 2 + x ** 2)
            b_ij = -x / (r ** 2 + x ** 2)

            ###### Va in label is pre-normalized by division over 50 (cf # Normalize angles in pandapower_graph.py)
            e_i = vm_i * torch.cos(va_i)
            f_i = vm_i * torch.sin(va_i)
            e_j = vm_j * torch.cos(va_j)
            f_j = vm_j * torch.sin(va_j)

            #### PowerflowNet ######
            Pji = g_ij * (e_i * e_j - e_i ** 2 + f_i * f_j - f_i ** 2) + b_ij * (f_i * e_j - e_i * f_j)
            Qji = g_ij * (f_i * e_j - e_i * f_j) + b_ij * (-e_i * e_j + e_i ** 2 - f_i * f_j + f_i ** 2)

        elif version=="2":
            r_x = torch.stack([r,x], dim=1)
            zm_ij = torch.norm(r_x, p=2, dim=-1, keepdim=True).squeeze(1) # (num_edges, 1) NOTE (r**2+x**2)**0.5 should be non-zero
            za_ij = torch.acos(r / zm_ij) # (num_edges, 1)
            ym_ij = 1/(zm_ij + 1e-6)        # (num_edges, 1)
            ya_ij = -za_ij      # (num_edges, 1)
            g_ij = ym_ij * torch.cos(ya_ij) # (num_edges, 1)
            b_ij = ym_ij * torch.sin(ya_ij) # (num_edges, 1)
            g_ij2 = r / (r ** 2 + x ** 2)
            b_ij2 = -x / (r ** 2 + x ** 2)

            Pij = g_ij * vm_i**2 - vm_i * vm_j * (g_ij * torch.cos(va_i - va_j) + b_ij * torch.sin(va_i - va_j))
            Qij = -b_ij * vm_i**2 - vm_i * vm_j * (g_ij * torch.sin(va_i - va_j) - b_ij * torch.cos(va_i - va_j)) - b * vm_i**2
            Pji = g_ij * vm_j**2 - vm_i * vm_j * (g_ij * torch.cos(va_j - va_i) + b_ij * torch.sin(va_j - va_i))
            Qji = -b_ij * vm_j**2 - vm_i * vm_j * (g_ij * torch.sin(va_j - va_i) - b_ij * torch.cos(va_j - va_i)) - b * vm_j**2
        ##### merging them by node i for each bus
        mask = mask.float()
        mask = mask.to(Pij.device)  # or mask.cuda()
        mask_t = mask_t.float()
        mask_t = mask_t.to(Pji.device)
        Pi = torch.matmul(mask, Pij) + torch.matmul(mask_t, Pji)
        Qi = torch.matmul(mask, Qij) + torch.matmul(mask_t, Qji)

        ## Second side of the loss

        bus_generator = torch.zeros_like(bus_features)
        bus_ext = torch.zeros_like(bus_features)

        # Step by step
        # predicted_gen_P = data.sn_mva[0] * out.get("gen")[gen_index, 0]
        # predicted_gen_Q = data.sn_mva[0] * out.get("gen")[gen_index, 1]
        # predicted_ext_P = data.sn_mva[0] * out.get("ext_grid")[ext_index, 2]
        # predicted_ext_Q = data.sn_mva[0] * out.get("ext_grid")[ext_index, 3]
        # bus_generator[bus_to_gen_index, 0] = predicted_gen_P
        # bus_generator[bus_to_gen_index, 1] = predicted_gen_Q
        # bus_ext[bus_to_ext_index, 0] = predicted_ext_P
        # bus_ext[bus_to_ext_index, 1] = predicted_ext_Q

        # Equivalent
        bus_generator[bus_to_gen_index, :] =  gen_buses
        bus_ext[bus_to_ext_index, :] = ext_buses

        bus_true = -bus_features/S_base + bus_generator + bus_ext
        Pi_true = bus_true[:, 0]
        Qi_true = bus_true[:, 1]

        duration = time.time() - begin
        loss = [mse_loss(Pi, Pi_true).unsqueeze(0), mse_loss(Qi, Qi_true).unsqueeze(0)]
        cat_loss = torch.cat(loss, dim=-1) / (out.get("bus").shape[0]**2)
        details = {"Pi":Pi,"Pi_true":Pi_true,"Qi":Qi,"Qi_true":Qi_true}

    return cat_loss, neighboorhood, duration, details
    #return cat_loss / S_base**2, neighboorhood, duration  # (num_edges, 2)


def power_cost_loss(data, out, neighboorhood=None, ground_truth=False):
    """calculate injected power Pji

    Formula:
    $$
    P_cost = cp0_eur + cp1_eur * P * + cp2_eur * P**2
    Q_cost = cp0_eur + cp1_eur * P * + cp2_eur * P**2
    $$

    Input:
    data: the GNN features and labels
    output: the ouput of the GCN, P and Q for genertors and external grid and Vm,Va for loads
    ground_truth: True to compute the losses on the ground truth, should return close to 0 (for debugging purpose)

    Return:
        MSE loss on Pi and Qi between the two terms of the loss

    """
    ground_truth = int(os.environ.get("USE_GT_POWERIMBALANCE", ground_truth))

    neighboorhood, S_base, mask, mask_t, bus_features, bus_to_line_list, line_to_bus_list, bus_to_gen_index, gen_index, bus_to_ext_index, ext_index, bus_to_bus_dict, i, j, mask, mask_t, r, x, b = get_loss_parameters(neighboorhood, data, out)

    gen_costs_p = data.x_dict.get("gen")[:, -6:-3]
    ext_costs_p= data.x_dict.get("ext_grid")[:, -6:-3]

    gen_costs_q = data.x_dict.get("gen")[:, -3:]
    ext_costs_q = data.x_dict.get("ext_grid")[:, -3:]

    if ground_truth:
        gen_buses = data["gen"].y[gen_index, :]
        ext_buses = data["ext_grid"].y[ext_index, :]
    else:
        gen_buses = out.get("gen")[:, 0:2]
        ext_buses = out.get("ext_grid")[:, 2:4]

    gen_cost_p = gen_costs_p[:,0] +  gen_costs_p[:,1] * gen_buses[:,0] + gen_costs_p[:,2] * torch.square(gen_buses[:,0])
    ext_cost_p = ext_costs_p[:, 0] + ext_costs_p[:, 1] * ext_buses[:, 0] + ext_costs_p[:, 2] * torch.square(ext_buses[:, 0])

    gen_cost_q = gen_costs_q[:, 0] + gen_costs_q[:, 1] * gen_buses[:, 1] + gen_costs_q[:, 2] * torch.square(gen_buses[:, 1])
    ext_cost_q = ext_costs_q[:, 0] + ext_costs_q[:, 1] * ext_buses[:, 1] + ext_costs_q[:, 2] * torch.square(ext_buses[:, 1])

    loss = torch.cat([gen_cost_p,ext_cost_p])

    return loss



