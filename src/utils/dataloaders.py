from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch.utils.data import ConcatDataset
from src.utils.transforms import ToTransformerInput

def prepare_dataloaders(case_name, train_limit, test_limit, dataset, dataroot, train_batch_size, batch_size,
                        eval_splits, serial_batches, n_train_examples=None, n_test_examples=None,train_case_name="",
                        layers="GraphConv"):
    train_ds = None
    test_ds = None
    generalization_ds = None
    train_limit = int(train_limit)
    test_limit = int(test_limit)

    print("loading dataset")
    case_names = case_name.split("+")
    train_case_names = train_case_name.split("+")

    train_ds_list = []
    test_ds_list = []
    generalization_ds_list = []
    transform = None

    if "SAGEConv" in layers or "GraphConv" in layers or "GATConv" in layers or "TransformerConv" in layers:
        transform = T.ToUndirected()
    elif "Transformer" in layers:
        projector_dimension = int(layers.split(":")[1]) if len(layers.split(":"))>1 else 64
        transform = ToTransformerInput(output_dim=projector_dimension)

    if len(train_case_name):
        for case_name in train_case_names:

            if dataset == "OPFDataset":
                train_ds = OPFDataset(dataroot, case_name=case_name, split='train', num_groups=train_limit,
                                      transform=transform) if train_limit else None
            else:
                raise NotImplementedError

            if train_ds is not None:
                train_ds = train_ds[:n_train_examples] if n_train_examples is not None else train_ds
                train_ds_list.append(train_ds)

    for case_name in case_names:

        if dataset == "OPFDataset":
            if "test" in eval_splits:
                test_ds = OPFDataset(dataroot, case_name=case_name, split='test', num_groups=test_limit,
                                     transform=transform)
                test_ds_list.append(test_ds)
            if "generalization" in eval_splits:
                generalization_ds = OPFDataset(dataroot, case_name=case_name, split='test', num_groups=test_limit,
                                               transform=transform
                                               , topological_perturbations=True)
                generalization_ds_list.append(generalization_ds)


        if test_ds is not None:
            test_ds = test_ds[:n_test_examples] if n_test_examples is not None else test_ds
            test_ds_list.append(test_ds)
        if generalization_ds is not None:
            generalization_ds = generalization_ds[:n_test_examples] if n_test_examples is not None else generalization_ds
            generalization_ds_list.append(generalization_ds)

    # Batch and shuffle.
    training_loader = DataLoader(ConcatDataset(train_ds_list), batch_size=train_batch_size,
                                 shuffle=not serial_batches) if len(train_ds_list) else None
    test_loader = DataLoader(ConcatDataset(test_ds_list), batch_size=batch_size, shuffle=not serial_batches) if len(test_ds_list) else None
    generalization_loader = DataLoader(ConcatDataset(generalization_ds_list), batch_size=batch_size,
                                       shuffle=not serial_batches) if len(generalization_ds_list) else None

    return train_ds, test_ds, generalization_ds, training_loader, test_loader, generalization_loader
