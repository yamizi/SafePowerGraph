import sys
sys.path.append(".")
import os
import numpy as np
import json
from comet_ml import Experiment
import torch
from src.utils.models import get_model
from src.utils.dataloaders import prepare_dataloaders
from src.utils.training import train_epoch
from src.utils.optimizers import get_optimizer

from src.configs.train_config import  TrainOptions

def run(epoch_start=0, epoch_end=5,train_batch_size=256, batch_size=256, use_pinn_loss=0, device="cuda",
        case_name="pglib_opf_case14_ieee",lr_plateau_patience=5,lr_factor=0.5,lr_step_size=10, min_lr_plateau=1e-5,
        layers="GraphConv", experiment=None, dataroot='./data',lr_policy="plateau",lr=0.001,dataset="OPFDataset",
        pinn_loss_type=1, model_ckp=None,train_limit=20, test_limit=20,eval_splits=None,
        train_case_name="", save_epoch_freq=10, continue_train=False, **kwargs):


    train_ds, _, _, training_loader, test_loader, generalization_loader = (
        prepare_dataloaders(case_name,train_limit,test_limit,dataset,dataroot, train_batch_size,
                            batch_size,eval_splits, kwargs.get("serial_batches"), train_case_name=train_case_name
                            , layers=layers
                            ))



    model_path = "output/models/model_{}".format(experiment.id) if model_ckp is None else model_ckp
    os.makedirs(model_path,exist_ok=True)

    # Initialise the model.
    # data.metadata() here refers to the PyG graph metadata, not the OPFData metadata.
    data = train_ds[0].to(device)

    # list files ending with .pt in the folder model_path

    model_ckp = os.path.join(model_path, "last.pt") if continue_train is None else None
    model = get_model(data=data, device=device, layers=layers, model_ckp=model_ckp)

    optimizer, lr_scheduler = get_optimizer(model, lr, lr_policy,lr_plateau_patience,lr_factor,lr_step_size,
                                            min_lr_plateau, path_ckp=model_ckp)

    with torch.no_grad(): # Initialize lazy modules.
        #model(data.x_dict, data.edge_index_dict)
        model.train()
        best_loss = np.inf

    # get current epoch from last json
    if continue_train and os.path.exists(os.path.join(model_path, "last.json")):
        with open(os.path.join(model_path, "last.json"), "r") as f:
            log = json.load(f)
            epoch_start = log["epoch"] + 1
            experiment.log_parameter("epoch_start", epoch_start)

    graph_representation = not layers.startswith("Transformer:")
    for epoch in range(epoch_start, epoch_end):
        val_loss = train_epoch(epoch, training_loader, test_loader, model, optimizer, lr_scheduler, device, experiment,
                    use_pinn_loss, pinn_loss_type, generalization_loader,graph_representation=graph_representation )
        if val_loss<best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_path, "best.pt"))
            experiment.log_model("best_model", os.path.join(model_path, "best.pt"))

        if (epoch + 1) % save_epoch_freq == 0:
            torch.save(model.state_dict(), os.path.join(model_path, "epoch_{}.pt".format(epoch)) )
            torch.save(model.state_dict(), os.path.join(model_path, "last.pt"))
            experiment.log_model("last_model", os.path.join(model_path, "last.pt"))
            log = {"epoch": epoch,"epoch_start": epoch_start, "epoch_end": epoch_end, "loss": val_loss}
            
            with open(os.path.join(model_path, "last.json"), "w") as f:
                json.dump(log, f)
            # save optimizer and lr_scheduler
            torch.save(optimizer.state_dict(), os.path.join(model_path, "optimizer_last.pt"))
            torch.save(lr_scheduler.state_dict(), os.path.join(model_path, "lr_scheduler_last.pt"))

    torch.save(model.state_dict(),os.path.join(model_path,"last.pt") )
    experiment.log_model("last_model", os.path.join(model_path,"last.pt"))
    experiment.end()


if __name__ == '__main__':
    options = TrainOptions()
    params, experiment = options.parse()
    run(experiment=experiment, **params)