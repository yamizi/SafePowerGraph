import torch
import os
def get_optimizer(model, lr, lr_policy,lr_plateau_patience=5,lr_factor=0.5,lr_step_size=10, min_lr_plateau=1e-5,
                  path_ckp=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if lr_policy == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_plateau_patience,
                                                                  factor=lr_factor, min_lr=min_lr_plateau)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_factor)


    if path_ckp is not None:
        optimizer_ckp = os.path.join(path_ckp, "optimizer_last.pt")
        lr_scheduler_ckp = os.path.join(path_ckp, "lr_scheduler_last.pt")

        if os.path.exists(optimizer_ckp):
            optimizer.load_state_dict(torch.load(optimizer_ckp))
        if os.path.exists(lr_scheduler_ckp):
            lr_scheduler.load_state_dict(torch.load(lr_scheduler_ckp))

    return optimizer, lr_scheduler