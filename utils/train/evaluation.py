import torch.nn.functional as F
import torch
import numpy as np
from utils.models import GNN
from utils.train.helpers import heterogeneous_loss_step, timeit

from utils import homo_to_hetero

def evaluate(model, val_loader, device, y_nodes, loss_fn, hetero, clamp_boundary, use_physical_loss, epoch, log_every,
             neighboorhood=None, node_types=None):
    val_loss = 0
    boundary_loss_val = 0
    physical_loss_val = 0
    cost_loss_val = 0
    val_loss_gen = 0
    val_loss_ext_grid = 0
    val_loss_bus = 0
    val_loss_line = 0

    val_losses_all = []
    out_all = []
    labels_all = []
    with torch.no_grad():
        for batch_id, batch in enumerate(val_loader):
            last_out, last_label, loss, losses, b_losses, p_losses, c_losses, neighboorhood = eval_step(model, batch.to(
                device), None,
                                                                                                        y_nodes,
                                                                                                        loss_fn,
                                                                                                        hetero,
                                                                                                        clamp_boundary=(
                                                                                                                clamp_boundary == 2 or clamp_boundary == 3),
                                                                                                        use_physical_loss=use_physical_loss,
                                                                                                        neighboorhood=neighboorhood,
                                                                                                        epoch=epoch,
                                                                                                        batch_id=batch_id,
                                                                                                        node_types=node_types)
            val_loss += loss
            boundary_loss_val += np.concatenate(b_losses, 0).mean() if len(b_losses) else np.array([0])

            ls = np.array([e.flatten() for e in p_losses]) if len(p_losses) else np.array([0])
            physical_loss_val += ls.mean()

            ls = np.array([e for e in c_losses]) if len(c_losses) else np.array([0])
            cost_loss_val += ls.mean()

            losses["boundary"] = np.array(b_losses)
            losses["physical"] = np.array(p_losses)
            losses["cost"] = np.array(c_losses)
            losses["weighted"] = np.array(loss)

            val_losses_all.append(losses)
            out_all.append(last_out)
            labels_all.append(last_label)
            val_loss_gen += losses["gen"].mean() if losses.get("gen", None) is not None else 0
            val_loss_ext_grid += losses["ext_grid"].mean() if losses.get("ext_grid", None) is not None else 0
            val_loss_bus += losses["bus"].mean() if losses.get("bus", None) is not None else 0
            val_loss_line += losses["line"].mean() if losses.get("line", None) is not None else 0

    val_loss /= len(val_loader)
    boundary_loss_val /= len(val_loader)
    physical_loss_val /= len(val_loader)
    val_loss_gen /= len(val_loader)
    val_loss_bus /= len(val_loader)
    val_loss_ext_grid /= len(val_loader)
    val_loss_line /= len(val_loader)

    if epoch % log_every == 0:
        print("validation loss", val_loss)
        print("boundary loss", boundary_loss_val)
        print("physical loss", physical_loss_val)
        print("cost loss", cost_loss_val)

    return val_loss, boundary_loss_val, physical_loss_val, cost_loss_val, val_loss_gen, val_loss_ext_grid, val_loss_bus, val_loss_line, out_all, labels_all, val_losses_all



def eval_step(model, data, mask_node="paper", feature_node="paper", loss_f=None, hetero=True, clamp_boundary=0,
              use_physical_loss="2", neighboorhood=None, epoch=0, batch_id=0, node_types=None):
    model.eval()
    if loss_f is None:
        loss_f = F.cross_entropy

    if isinstance(feature_node, str):
        feature_node = [feature_node]
    if isinstance(mask_node, str):
        mask_node = [mask_node]
    loss = 0

    boundary_losses = []
    physical_losses = []
    cost_losses = []

    return_label = {}
    return_output = {}
    use_physical_loss = use_physical_loss.split("_")

    if hasattr(model,"first_conv"):
        out_ = model(data.x_dict if hetero else data.x, data.edge_index_dict if hetero else data.edge_index, None)
    else:
        out_ = model(data)

    if not hetero:
        data, out = homo_to_hetero(data, out_, node_types)
    else:
        out = out_

    
    timeit(model, data)
    return_output, return_label, loss, losses, boundary_losses, physical_losses, cost_losses = heterogeneous_loss_step(
        feature_node, data, out, loss, loss_f, clamp_boundary, return_output, return_label,
        {}, boundary_losses, use_physical_loss, physical_losses, cost_losses, mask_node,
        neighboorhood=neighboorhood)


    return return_output, return_label, float(
        loss), losses, boundary_losses, physical_losses, cost_losses, neighboorhood