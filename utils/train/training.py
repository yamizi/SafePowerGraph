import time
from argparse import Namespace

import torch.nn.functional as F
import torch
import numpy as np
from utils.models import GNN
from utils.train.helpers import clamp_boundaries
from utils.train.evaluation import evaluate
import os
from utils.losses import boundary_loss, node_loss, power_imbalance_loss, power_cost_loss




def train_opf(model, train_loader, val_loader, max_epochs=200, y_nodes=["gen", "ext_grid"], node_types=["bus","load","gen", "ext_grid"],log_every=10,
              device="cpu", decay_lr=0.3, hetero=True, base_lr=0.01, experiment=None, clamp_boundary=0,
              use_physical_loss=1, weighting="relative"):
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    milestones = [max_epochs // 2, (max_epochs * 3) // 4, (max_epochs * 9) // 10]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=decay_lr)

    print("Training model with device", next(model.parameters()).device)

    loss_fn = node_loss

    train_losses = []
    train_losses_bus = []
    train_losses_line = []
    train_losses_gen = []
    train_losses_ext_grid = []

    boundary_train_losses = []
    physical_train_losses = []
    cost_train_losses = []
    boundary_val_losses = []
    physical_val_losses = []
    cost_val_losses = []
    val_losses = []
    val_losses_bus = []
    val_losses_line = []
    val_losses_gen = []
    val_losses_ext_grid = []
    learning_rate = []

    neighboorhood = None
    nb_epoch_profiling = 5
    prof = None
    if os.environ.get("LOG_TraceAnalysis", ""):
        profile_path = os.environ.get("LOG_TraceAnalysis", "")
        os.makedirs(profile_path, exist_ok=True)
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3,
                                             repeat=(len(train_loader) // 5) * nb_epoch_profiling),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path),
            record_shapes=True,
            with_stack=True)
        prof.start()

    if max_epochs==0:
        print("No training, evaluation only")

        val_loss, boundary_loss_val, physical_loss_val, cost_loss_val, val_loss_gen, val_loss_ext_grid, val_loss_bus, val_loss_line, out_all, labels_all, val_losses_all = evaluate(
            model, val_loader, device, y_nodes, loss_fn, hetero, clamp_boundary, use_physical_loss, 0,
            log_every)

        val_losses.append(val_loss)
        boundary_val_losses.append(boundary_loss_val)
        physical_val_losses.append(physical_loss_val)
        cost_val_losses.append(cost_loss_val)
        val_losses_gen.append(val_loss_gen)
        val_losses_ext_grid.append(val_loss_ext_grid)
        val_losses_bus.append(val_loss_bus)
        val_losses_line.append(val_loss_line)

        if experiment is not None:
            log_dict = {"p_val_losses": physical_loss_val, "c_val_losses": cost_loss_val,
                        "b_val_losses": boundary_loss_val,
                        "val_losses_gen": val_loss_gen, "val_losses_ext_grid": val_loss_ext_grid,
                        "val_losses_bus": val_loss_bus, "val_losses_line": val_loss_line}

            experiment.log_metrics(log_dict, epoch=0)
            logged_metrics = {}
            outputs_keys = {"gen": [0, 1], "ext_grid": [2, 3], "bus": [4, 5]}

            if isinstance(out_all, dict):
                for k, v in out_all[0].items():
                    logged_metrics = {**logged_metrics,
                                      **{f"out_{k}_{i}_0": val.cpu().item() for (i, val) in enumerate(v[0]) if
                                         i in outputs_keys.get(k)}}
                    logged_metrics = {**logged_metrics,
                                      **{f"out_{k}_{i}_1": val.cpu().item() for (i, val) in enumerate(v[1]) if
                                         i in outputs_keys.get(k)}}

                    logged_metrics = {**logged_metrics,
                                      **{f"label_{k}_{i}_0": val.cpu().item() for (i, val) in
                                         enumerate(labels_all[0][k][0])}}
                    logged_metrics = {**logged_metrics,
                                      **{f"label_{k}_{i}_1": val.cpu().item() for (i, val) in
                                         enumerate(labels_all[0][k][1])}}

                experiment.log_metrics(logged_metrics, epoch=0)
    else:
        for epoch in range(0, max_epochs):
            lr = lr_scheduler.get_last_lr()[0]
            learning_rate.append(lr)
            ssl_weight = 0
            train_loss = 0
            boundary_train_loss = 0
            physical_train_loss = 0
            cost_train_loss = 0
            physical_train_loss_duration = []

            train_loss_gen = 0
            train_loss_ext_grid = 0
            train_loss_bus = 0
            train_loss_line = 0
            print("epoch ",epoch)

            for batch_id, batch in enumerate(train_loader):
                if prof is not None and epoch < nb_epoch_profiling:
                    prof.step()
                out, labels, loss, losses, b_losses, p_losses, cost_losses, neighboorhood, physical_loss_duration, ssl_weight = train_step(
                    model, optimizer, batch.to(device), None,
                    y_nodes, loss_fn, hetero,
                    clamp_boundary=(
                            clamp_boundary == 1 or clamp_boundary == 2),
                    use_physical_loss=use_physical_loss,
                    neighboorhood=neighboorhood,
                    epoch=epoch, batch_id=batch_id,
                    weighting=weighting, max_epochs=max_epochs,
                node_types=node_types)
                train_loss += torch.cat(losses, 0).mean().cpu().item()

                train_loss_gen += losses[0].mean().cpu().item()
                train_loss_ext_grid += losses[1].mean().cpu().item() if len(losses) > 1 else 0
                train_loss_bus += losses[2].mean().cpu().item() if len(losses) > 2 else 0
                train_loss_line += losses[3].mean().cpu().item() if len(losses) > 3 else 0

                boundary_train_loss += np.concatenate([e.cpu().detach().numpy() for e in b_losses], 0).mean() if len(
                    b_losses) else np.array([0])

                list_p = [e.cpu().detach().numpy().flatten() for e in p_losses] if len(p_losses) else [np.array([0])]
                ls = np.concatenate(list_p)
                physical_train_loss += ls.mean().mean()
                physical_train_loss_duration = physical_train_loss_duration + physical_loss_duration

                ls = np.array([e.cpu().detach().numpy().flatten() for e in cost_losses]) if len(cost_losses) else np.array(
                    [0])
                cost_train_loss += ls.mean().mean()

            lr_scheduler.step()
            train_loss /= len(train_loader)
            boundary_train_loss /= len(train_loader)
            physical_train_loss /= len(train_loader)
            cost_train_loss /= len(train_loader)

            train_losses_gen.append(train_loss_gen / len(train_loader))
            train_losses_ext_grid.append(train_loss_ext_grid / len(train_loader))
            train_losses_bus.append(train_loss_bus / len(train_loader))
            train_losses_line.append(train_loss_line / len(train_loader))

            if epoch % log_every == 0:
                print("epoch", epoch)
                print("training loss", train_loss)
                print("boundary loss", boundary_train_loss)
                print("physical loss", physical_train_loss)
                print("ssl_weight", ssl_weight)
                print("cost loss", cost_train_loss)

            train_losses.append(train_loss)
            boundary_train_losses.append(boundary_train_loss)
            physical_train_losses.append(physical_train_loss)
            cost_train_losses.append(cost_train_loss)

            val_loss, boundary_loss_val, physical_loss_val, cost_loss_val, val_loss_gen, val_loss_ext_grid, val_loss_bus, val_loss_line, out_all, labels_all, val_losses_all = evaluate(
                model, val_loader, device, y_nodes, loss_fn, hetero, clamp_boundary, use_physical_loss, epoch,
                log_every, neighboorhood=neighboorhood)

            val_losses.append(val_loss)
            boundary_val_losses.append(boundary_loss_val)
            physical_val_losses.append(physical_loss_val)
            cost_val_losses.append(cost_loss_val)
            val_losses_gen.append(val_loss_gen)
            val_losses_ext_grid.append(val_loss_ext_grid)
            val_losses_bus.append(val_loss_bus)
            val_losses_line.append(val_loss_line)

            if experiment is not None:
                log_dict = {"train_losses": train_loss, "ssl_weight": ssl_weight,
                            "train_losses_gen": train_loss_gen, "train_losses_ext_grid": train_loss_ext_grid,
                            "train_losses_bus": train_loss_bus,
                            "val_losses": val_loss, "p_train_losses": physical_train_loss,
                            "p_train_losses_duration": np.sum(physical_train_loss_duration),
                            "p_val_losses": physical_loss_val, "c_val_losses": cost_loss_val,
                            "b_train_losses": boundary_train_loss, "b_val_losses": boundary_loss_val, "learning_rate": lr,
                            "val_losses_gen": val_loss_gen, "val_losses_ext_grid": val_loss_ext_grid,
                            "val_losses_bus": val_loss_bus, "val_losses_line": val_loss_line}
                experiment.log_metrics(log_dict, epoch=epoch)
                logged_metrics = {}
                outputs_keys = {"gen": [0, 1], "ext_grid": [2, 3], "bus": [4, 5]}

                if isinstance(out_all, dict):
                    for k, v in out_all[0].items():
                        logged_metrics = {**logged_metrics,
                                          **{f"out_{k}_{i}_0": val.cpu().item() for (i, val) in enumerate(v[0]) if
                                             i in outputs_keys.get(k)}}
                        logged_metrics = {**logged_metrics,
                                          **{f"out_{k}_{i}_1": val.cpu().item() for (i, val) in enumerate(v[1]) if
                                             i in outputs_keys.get(k)}}

                        logged_metrics = {**logged_metrics,
                                          **{f"label_{k}_{i}_0": val.cpu().item() for (i, val) in
                                             enumerate(labels_all[0][k][0])}}
                        logged_metrics = {**logged_metrics,
                                          **{f"label_{k}_{i}_1": val.cpu().item() for (i, val) in
                                             enumerate(labels_all[0][k][1])}}

                    experiment.log_metrics(logged_metrics, epoch=epoch)

        print("Training over")
    return train_losses, val_losses, (val_losses_gen, val_losses_ext_grid, val_losses_bus, val_losses_line), (
        out_all, val_losses_all), boundary_train_losses, physical_train_losses, boundary_val_losses, learning_rate


def train_step(model, optimizer, batch, mask_node="paper", feature_node="paper", loss_f=None, hetero=True,
               use_boundary_loss=True, clamp_boundary=True, use_physical_loss="2", neighboorhood=None
               , epoch=0, batch_id=0, weighting="relative", max_epochs=100, node_types=[]):
    data = batch.clone()
    model.train()
    optimizer.zero_grad()
    if loss_f is None:
        loss_f = F.mse_loss

    if isinstance(feature_node, str):
        feature_node = [feature_node]
    if isinstance(mask_node, str):
        mask_node = [mask_node]

    loss = 0
    ssl_weight = 0
    losses = []
    boundary_losses = []
    physical_losses = []
    cost_losses = []
    physical_losses_duration = []
    return_label = {}
    return_output = {}

    use_physical_loss = use_physical_loss.split("_")

    ssl_start_epoch = int(use_physical_loss[2]) if len(use_physical_loss) > 2 else max_epochs // 4
    ssl_max_value = float(use_physical_loss[3]) if len(use_physical_loss) > 3 else 1
    ssl_power = int(use_physical_loss[4]) if len(use_physical_loss) > 4 else 2


    if not hetero:
        out_homo = model(data)
        hetero_dict = {}
        out = {}
        for node_type_id, node_type_name in enumerate(node_types):
            mask = (data.node_type == node_type_id)
            hetero_dict[node_type_name] = Namespace(x=data.x[mask],y=data.y[mask],boundaries=data.boundaries[mask])
            out[node_type_name] = out_homo[mask]
        data = hetero_dict
    else:
        out = model(data.x_dict, data.edge_index_dict)

    total_nodes = {node: len(data[node].y) for node in feature_node}
    # out = denormalize_outputs(out,data.sn_mva[0])
    for i, node in enumerate(feature_node):
        label = data[node].y
        output = out[node]
        if mask_node is not None:
            mask = data[mask_node[i]].train_mask
            label = label[mask_node[i]]
            output = output[mask_node[i]]

        output_mask = data[node].output_mask

        loss_label = loss_f(label, output, output_mask)

        if clamp_boundary:
            output = clamp_boundaries(data[node].boundaries, output, node, output_mask)

        return_output[node] = output
        return_label[node] = label

        if use_boundary_loss:
            loss_boundary = boundary_loss(data[node].boundaries, output, node=node)
            loss_node = torch.cat([loss_label.reshape((loss_boundary.shape[0], -1)), loss_boundary.unsqueeze(1)], 1)
            boundary_losses.append(loss_boundary)
        else:
            loss_node = loss_label

        weight_node = np.sum(list(total_nodes.values())) / total_nodes.get(node) if "relative" in weighting else 1
        loss += weight_node * loss_node.mean()
        losses.append(weight_node * loss_node)

        if use_physical_loss[0] != "0" and node == "bus":

            use_physical_loss_version = use_physical_loss[1] if len(use_physical_loss) == 2 else "2"
            physical_loss, neighboorhood, duration, _ = power_imbalance_loss(data, out, neighboorhood,
                                                                          ground_truth=False,
                                                                          version=use_physical_loss_version)

            cost_loss = power_cost_loss(data, out, neighboorhood, ground_truth=False)
            physical_losses_duration.append(duration)
            cost_losses.append(weight_node * cost_loss.mean())

            if int(use_physical_loss[0]) == 2:
                loss += weight_node * physical_loss.mean()
                physical_losses.append(weight_node * physical_loss.mean())
            elif int(use_physical_loss[0]) == 3:
                loss = weight_node * physical_loss.mean()
                physical_losses.append(weight_node * physical_loss.mean())

            if int(use_physical_loss[0]) == 21:
                loss += weight_node * (physical_loss.mean() + cost_loss.mean())
                physical_losses.append(weight_node * physical_loss.mean())

            elif int(use_physical_loss[0]) == 31:
                loss = weight_node * (physical_loss.mean() + cost_loss.mean())
                physical_losses.append(weight_node * physical_loss.mean())
            else:
                physical_losses.append(physical_loss)

    # else:
    #     mask = ~torch.isnan(data.y)
    #     label = data.y
    #     if isinstance(model, GNN):
    #         out = model(data)
    #         # out = denormalize_outputs(out, data.sn_mva[0],data.angle[0], "gnn")
    #     else:
    #         x = data.x.reshape(data.batch_size, -1)
    #         label = label.reshape(data.batch_size, -1)
    #         out = model(x)
            # out = denormalize_outputs(out, data.sn_mva[0],data.angle[0], "fcnn")

        return_output = out[mask]
        loss_label = loss_f(label[mask], out, mask)
        # loss_boundary = boundary_loss(data[node].boundaries, output)
        loss_node = loss_label  # torch.cat([loss_label,loss_boundary.unsqueeze(1)],1)
        losses.append(loss_label)
        # boundary_losses.append(loss_boundary.cpu().detach().numpy())
        loss += loss_node.mean()

    if "random" in weighting:
        ls = physical_losses + boundary_losses + losses
        all_losses = torch.stack([l.mean() for l in ls])
        random_weights = torch.nn.functional.softmax(torch.rand(len(all_losses))).to(all_losses.device)
        loss = torch.dot(random_weights, all_losses)

    elif "sup2ssl" in weighting:
        ssl_weight = np.min(
            [ssl_max_value, (np.max([0, epoch - ssl_start_epoch]) / (max_epochs - ssl_start_epoch)) ** ssl_power])
        loss = ssl_weight * torch.cat(losses, 0).mean() + (1 - ssl_weight) * torch.cat(physical_losses, 0).mean()

    elif "sup2cost" in weighting:
        ssl_weight = np.min(
            [ssl_max_value, (np.max([0, epoch - ssl_start_epoch]) / (max_epochs - ssl_start_epoch)) ** ssl_power])
        loss = ssl_weight * torch.cat(losses, 0).mean() + (1 - ssl_weight) * torch.cat(cost_losses, 0).mean()

    elif "sup2sslcost" in weighting:
        ssl_weight = np.min(
            [ssl_max_value, (np.max([0, epoch - ssl_start_epoch]) / (max_epochs - ssl_start_epoch)) ** ssl_power])
        loss = ssl_weight * torch.cat(losses, 0).mean() + (1 - ssl_weight) * torch.cat(cost_losses + physical_losses,
                                                                                       0).mean()

    loss.backward()
    optimizer.step()

    # .cpu().detach().numpy()

    return return_output, return_label, float(
        loss), losses, boundary_losses, physical_losses, cost_losses, neighboorhood, physical_losses_duration, ssl_weight





