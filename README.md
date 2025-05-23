### License:

MIT License

## Documentation

Extensive documentation is provided in https://ballistic-purple-6f5.notion.site/SafePowerGraph-Documentation-1fcbf710ca5980fdafccccf2a9891fc3

## Examples

the file "example_script.sh" provides a list of different commands with explanation of the parameters;



## Third party tools:

### Trackers:
To enable experiment tracking with comet (www.comet.com) set the environment variable `COMET_APIKEY`

`export COMET_APIKEY=XXX`

### Models:

We provide A model zoo of pretrained models using different weighting, architecture, and pinn strategies
They are organized by grid size and available on: https://figshare.com/projects/SafePowerGraph/248768


### Datasets:

- To enable Hugging Face to load OUR datasets, set the environment variable `HUGGINFACE_TOKEN` 

`export HUGGINFACE_TOKEN=XXX`

Our datasets are Licensed under the Creative Commons Attribution 4.0

- We support the OPF Dataset in our benchmark https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.OPFDataset.html#torch_geometric.datasets.OPFDataset  with grids up to 13659 bus
to use them, train and evaluate the models using the script in experiments/train_eval_largeOPF.py

This dataset is also licensed under the Creative Commons Attribution 4.0

For more information on this dataset, refer to their README: https://storage.mtls.cloud.google.com/gridopt-dataset/README