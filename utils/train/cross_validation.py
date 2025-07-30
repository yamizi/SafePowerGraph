import torch.nn.functional as F
import torch
import numpy as np
from utils.models import GNN
from torch_geometric.nn import to_hetero
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from torch_geometric.loader import DataLoader
import pickle
from utils.losses import node_loss

from utils.train.training import train_step
from utils.train.evaluation import eval_step
from utils.train.helpers import auto_garbage_collect

def objective(config, graph, device, max_epochs, y_nodes, hetero, train_loader, val_loader, clamp_boundary,
              use_physical_loss=1):
    # print("objective", config)  # ①

    model = GNN(hidden_channels=[config.get("hidden_channels") for i in range(config.get("nb_hidden_layers"))],
                out_channels=graph.num_outputs, aggr=config.get("aggr"), cls=config.get("cls"))

    metadata = graph[0].cpu().metadata()
    model = to_hetero(model, metadata, aggr='sum')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr"))
    milestones = [max_epochs // 2, (max_epochs * 3) // 4, (max_epochs * 9) // 10]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones,
                                                        gamma=config.get("decay_lr"))
    neighboorhood = None
    while True:
        for batch in train_loader:
            # return_output, return_label, float(loss), losses, boundary_losses, physical_losses, (
            # neighboorhood, physical_losses_duration), ssl_weight

            out, labels, loss, losses, b_losses, physical_losses, cost_losses, neighboorhood, physical_losses_duration, ssl_weight = train_step(
                model, optimizer, batch, None,
                y_nodes,
                node_loss, hetero,
                clamp_boundary=(
                        clamp_boundary == 1 or clamp_boundary == 2),
                use_physical_loss=use_physical_loss,
                neighboorhood=neighboorhood)
        lr_scheduler.step()

        val_loss_gen = 0
        val_loss_ext_grid = 0
        val_loss_bus = 0
        val_loss_line = 0

        for batch in val_loader:
            with torch.no_grad():
                last_out, last_label, loss, losses, b_losses, p_losses, c_losses, neighboorhood = eval_step(model,
                                                                                                            batch, None,
                                                                                                            y_nodes,
                                                                                                            node_loss,
                                                                                                            hetero,
                                                                                                            clamp_boundary=(
                                                                                                                    clamp_boundary == 2 or clamp_boundary == 3),
                                                                                                            neighboorhood=neighboorhood,
                                                                                                            use_physical_loss=use_physical_loss)

            val_loss_gen += losses[0].mean()
            val_loss_ext_grid += losses[1].mean() if len(losses) > 1 else 0
            val_loss_bus += losses[2].mean() if len(losses) > 2 else 0
            val_loss_line += losses[3].mean() if len(losses) > 3 else 0

        auto_garbage_collect()

        ray.train.report({"val_loss_gen": val_loss_gen, "val_loss_ext_grid": val_loss_ext_grid,
                          "val_loss_bus": val_loss_bus, "val_loss_line": val_loss_line})  # Report to Tune



def train_cv(pickle_file, cv_ratio, graph, max_epochs=20, num_samples=10, y_nodes=["gen", "ext_grid", "bus"],
             device="cpu", hetero=True, base_lr=[0.1, 1], plot=False, num_graphs=1000, train_batch_size=32,
             val_batch_size=32, train_graphs=None, clamp_boundary=0, use_physical_loss=1):
    if train_graphs is None:
        with open(pickle_file, "rb") as pickled:
            print("Loading cross validation dataset from", pickle_file)
            loaded = pickle.load(pickled)

        train_graphs = loaded.get("train_graphs")
    train_list = [train_graphs[i][0].to(device) for i in range(num_graphs)]

    del train_graphs

    cv_list = train_list[:min(len(train_list), 1000)]
    nb_train = int(cv_ratio * len(cv_list))
    cv_train = cv_list[nb_train:]
    training_loader = DataLoader(cv_train, batch_size=val_batch_size)
    cv_val = cv_list[:nb_train]
    validation_loader = DataLoader(cv_val, batch_size=train_batch_size)

    lr_space = np.logspace(np.round(np.log(base_lr[0]) / np.log(10)), np.round(np.log(base_lr[1]) / np.log(10)), 3, 10)
    lr_decay = np.linspace(0.1, 0.9, 5)

    search_space = {"lr": tune.choice(lr_space.tolist()), "decay_lr": tune.choice(lr_decay.tolist()),
                    "hidden_channels": tune.choice([32, 64, 128, 256]), "nb_hidden_layers": tune.choice([1, 2, 3, 4]),
                    "aggr": tune.choice(["mean", "max"]), "cls": tune.choice(["gcn", "sage", "gat"])}

    print("running optuna search on ", search_space, "for epochs", max_epochs, "and size", len(training_loader.dataset)
          , "using device", device)

    metrics = ["val_loss_bus", "val_loss_gen", "val_loss_ext_grid"]
    # metrics = ["val_loss_gen", "val_loss_ext_grid"]
    algo = OptunaSearch(metric=metrics, mode=["min"] * len(metrics))  # ②

    # ray.init(num_cpus=10)

    tuner = tune.Tuner(  # ③
        tune.with_parameters(objective, graph=ray.put(graph), device=device, max_epochs=max_epochs, y_nodes=y_nodes,
                             hetero=hetero, train_loader=ray.put(training_loader),
                             val_loader=ray.put(validation_loader), clamp_boundary=clamp_boundary,
                             use_physical_loss=use_physical_loss),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=num_samples,
        ),
        run_config=ray.train.RunConfig(
            stop={"training_iteration": max_epochs},
        ),
        param_space=search_space
    )
    results = tuner.fit()
    dfs = {result.path.split("/")[-1]: result.metrics_dataframe for result in results}
    best_result = results.get_best_result("val_loss_gen", "min")
    print("Best config is:", best_result.metrics, best_result.config)

    if plot:
        ax = None  # This plots everything on the same plot
        for d in dfs.values():
            ax = d.val_loss_gen.plot(ax=ax, legend=False, logy=True)
        # ax.set_ylim([-10, 10])
        import matplotlib.pyplot as plt
        plt.show()

    return results.get_best_result("val_loss_gen", "min").config, dfs
