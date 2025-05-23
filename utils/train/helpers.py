import torch
import psutil, gc
from utils.losses import boundary_loss, power_imbalance_loss, power_cost_loss

def clamp_boundaries(boundaries, y, node=None, mask=None):
    if mask is not None:
        masked_y = y[mask.bool()].reshape((y.shape[0], -1))
    else:
        masked_y = y

    min_boundaries = torch.stack([boundaries[:, 2 * i] for i in range(y.shape[1])
                                  if torch.isnan(boundaries[:, 2 * i]).sum() == 0], 1)

    max_boundaries = torch.stack([boundaries[:, 2 * i + 1] for i in range(y.shape[1])
                                  if torch.isnan(boundaries[:, 2 * i + 1]).sum() == 0], 1)

    clamped_y = torch.clamp(masked_y, min_boundaries, max_boundaries)

    clamped = y * (1 - mask)
    clamped[mask.bool()] = clamped_y.flatten()
    return clamped


def auto_garbage_collect(pct=50.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return


def timeit(model, data, count=100000, hetero=True):
    import time
    return
    nb = count // len(data)
    params = (data.x_dict, data.edge_index_dict) if hetero else (data.x, data.edge_index)
    begin = time.time()

    for a in range(nb):
        model(*params)
    total = time.time() - begin
    print(total)


def heterogeneous_loss_step(feature_node, data, out, loss, loss_f, clamp_boundary, return_output, return_label,
                            losses, boundary_losses, use_physical_loss, physical_losses, cost_losses, mask_node=None,
                            neighboorhood=None):
    for i, node in enumerate(feature_node):
        label = data[node].y
        output = out[node]
        if mask_node is not None:
            mask = data[mask_node[i]].train_mask
            label = label[mask_node[i]]
            output = output[mask_node[i]]

        output_mask = data[node].output_mask
        loss_node = loss_f(label, output, output_mask)
        loss_boundary = boundary_loss(data[node].boundaries, output, node=node)

        if clamp_boundary:
            return_output[node] = clamp_boundaries(data[node].boundaries, output, node, output_mask)
        else:
            return_output[node] = output

        return_label[node] = label
        losses[node] = (loss_node.cpu().detach().numpy())
        boundary_losses.append(loss_boundary.cpu().detach().numpy())
        loss += loss_node.mean()

        if use_physical_loss[0] != "0" and node == "bus":
            use_physical_loss_version = use_physical_loss[1] if len(use_physical_loss) > 2 else "2"
            physical_loss, neighboorhood, duration, _ = power_imbalance_loss(data, out, neighboorhood=neighboorhood,
                                                                          version=use_physical_loss_version)
            physical_losses.append(physical_loss.mean().cpu().detach().numpy())

            cost_loss = power_cost_loss(data, out, neighboorhood, ground_truth=False)
            cost_losses.append(cost_loss.mean().cpu().detach().numpy())

    return return_output, return_label, loss, losses, boundary_losses, physical_losses, cost_losses