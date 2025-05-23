from matplotlib import pyplot as plt
import torch


def plot_losses(train_losses, val_losses, val_losses_gen, val_losses_ext_grid, case_name, title, save_path):
    # plt.close()
    # plt.cla()
    # plt.clf()

    # fig = plt.figure()
    plt.figure().clear()

    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="validation loss")
    plt.plot(val_losses_gen, label="validation loss generators")
    plt.plot(val_losses_ext_grid, label="validation loss ext_grid")

    plt.xlabel("training epoch")
    plt.ylabel("Error")
    plt.title('p_mw and q_mvar on ' + case_name + " " + title)
    plt.legend()
    plt.savefig(save_path + "/losses.png")
    print("over")


def plot_results(networks, val_graphs, outputs, y_nodes, constrained_networks, errors_network, case_name, title,
                 save_path):
    (out_all, val_losses_all) = outputs
    output_nodes = {node: torch.cat([e[node] for e in out_all], 0) for node in y_nodes}
    nb_gens = {node: len(networks.get("original")[node]) for node in y_nodes}

    for node, outputs in output_nodes.items():
        ground_truth = torch.cat([e.data[node].y for e in val_graphs])
        # fig = plt.figure()
        plt.figure().clear()

        plt.plot(outputs[:, 0].cpu(), label="Predicted P_mw")
        plt.plot(ground_truth[:, 0].cpu(), label="True P_mw")

        plt.plot(outputs[:, 1].cpu(), label="Predicted Q_mvar")
        plt.plot(ground_truth[:, 1].cpu(), label="True Q_mvar")

        plt.xlabel("Graphs")
        plt.ylabel("Predicted values")
        plt.title('{} on {}-{}'.format(node, case_name, title))
        plt.legend()
        plt.savefig(save_path + "/outputs-{}.png".format(node))
