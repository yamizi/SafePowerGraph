
from src.utils.losses import calculate_mismatch_batch, calculate_boundary_violations
import numpy as np
import torch.nn.functional as F
import torch
from tqdm import tqdm

def compute_pinn_loss(data,pinn_loss_type):
    powerflow_violations_loss = 0
    boundary_violations_loss = 0

    if pinn_loss_type == 1 or pinn_loss_type == 2:
        powerflow_violations = calculate_mismatch_batch(data)

        #Only consider P inbalance
        powerflow_vio= [torch.flatten(a[0]) for a in powerflow_violations]
        powerflow_violations_loss = torch.concat(powerflow_vio).square().sum().sqrt()
                                     # + torch.stack([a[1] for a in powerflow_violations]).square().sum().sqrt()

    if pinn_loss_type == 1 or pinn_loss_type == 3:
        boundary_violations = calculate_boundary_violations(data)
        values = [torch.concat(v).sum() for a,v in boundary_violations.items()]
        boundary_violations_loss = torch.stack(values).sum()

    return powerflow_violations_loss, boundary_violations_loss


def eval_batch(data, model, use_pinn_loss, pinn_loss_type, epoch, i, device, experiment, graph_representation=True,
               prefix="test"):

    with torch.no_grad():
        model.eval()
        data = data.to(device)
        out = model(data.x_dict, data.edge_index_dict) if graph_representation else model(data)
        loss_gen = F.mse_loss(out['generator'], data['generator'].y)
        loss_bus = F.mse_loss(out['bus'], data['bus'].y)
        experiment.log_metric(f"{prefix}_loss_gen", loss_gen, epoch=epoch, step=i)
        experiment.log_metric(f"{prefix}_loss_bus", loss_bus, epoch=epoch, step=i)
        val_loss = loss_gen + loss_bus
        if use_pinn_loss==1 or use_pinn_loss==3:
            ### PINN erorrs, can be included in a loss
            powerflow_violations, boundary_violations = compute_pinn_loss(data,pinn_loss_type)
            powerflow_violations = powerflow_violations / len(data)
            boundary_violations = boundary_violations / len(data)
            val_loss += (powerflow_violations + boundary_violations)
            experiment.log_metric(f"{prefix}_powerflow_violations", powerflow_violations, epoch=epoch, step=i)
            experiment.log_metric(f"{prefix}_boundary_violations", boundary_violations, epoch=epoch, step=i)

        return val_loss.item()

def train_epoch(epoch, training_loader, test_loader, model, optimizer, lr_scheduler, device, experiment, use_pinn_loss, 
                pinn_loss_type, generalization_loader=None, graph_representation=True):
    train_losses = []
    val_losses = []
    generalization_losses = []
    model.train()
    print("Training")
    for i, data in enumerate(tqdm(training_loader)):

        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x_dict, data.edge_index_dict) if graph_representation else model(data)
        loss_gen = F.mse_loss(out['generator'], data['generator'].y)
        loss_bus = F.mse_loss(out['bus'], data['bus'].y) #64 30 2

        experiment.log_metric("loss_gen", loss_gen, epoch=epoch, step=i)
        experiment.log_metric("loss_bus", loss_bus, epoch=epoch, step=i)
        loss = loss_gen + loss_bus
        ### PINN errors, can be included in a loss

        if use_pinn_loss==2 or use_pinn_loss==3:
            powerflow_violations, boundary_violations = compute_pinn_loss(data,pinn_loss_type)
            powerflow_violations = powerflow_violations / len(data)
            boundary_violations = boundary_violations / len(data)
            loss += powerflow_violations + boundary_violations
            experiment.log_metric("train_powerflow_violations", powerflow_violations, epoch=epoch, step=i)
            experiment.log_metric("test_boundary_violations", boundary_violations, epoch=epoch, step=i)

        loss.backward()
        experiment.log_metric("loss", loss, epoch=epoch, step=i)
        train_losses.append(loss.item())
        optimizer.step()

    print("Validation")
    model.eval()
    for i, data in enumerate(tqdm(test_loader)):
        val_loss = eval_batch(data, model, use_pinn_loss, pinn_loss_type, epoch, i, device, experiment,
                              graph_representation=graph_representation, prefix="test")
        experiment.log_metric("test_loss", loss, epoch=epoch, step=i)
        val_losses.append(val_loss)


    if generalization_loader:
        print("Generalization")
        for i, data in enumerate(tqdm(generalization_loader)):
            gen_loss = eval_batch(data, model, use_pinn_loss, pinn_loss_type, epoch, i, device, experiment, prefix="robust")
            experiment.log_metric("robust_loss", loss, epoch=epoch, step=i)
            generalization_losses.append(gen_loss)

    lr_scheduler.step(np.mean(val_losses))
    last_lr = lr_scheduler.get_last_lr() if lr_scheduler.get_last_lr() is not None else 0.0
    experiment.log_metric("lr", last_lr[0], epoch=epoch)
    print(f"Epoch {epoch}: Train loss {np.mean(train_losses)} - Val loss {np.mean(val_losses)} - Generalization loss {np.mean(generalization_losses)}  - LR: {last_lr}")
    return np.mean(val_losses)