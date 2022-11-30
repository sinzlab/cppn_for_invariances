import torch
import torch.nn as nn
import torch.nn as nn
from nnfabrik.builder import get_data, get_model
from invariance_generation.utils import extract_data_key
import glob


class ensamble(nn.Module):
    def __init__(
        self,
        submodels_fn="sensorium.models.stacked_core_full_gauss_readout",
        submodels_config={
            "pad_input": False,
            "stack": -1,
            "layers": 4,
            "input_kern": 9,
            "gamma_input": 6.3831,
            "gamma_readout": 0.0076,
            "hidden_dilation": 1,
            "hidden_kern": 7,
            "hidden_channels": 64,
            "depth_separable": True,
            "init_sigma": 0.1,
            "init_mu_range": 0.3,
            "gauss_type": "full",
        },
        dataset_fn="sensorium.datasets.static_loaders",
        dataset_config={
            "paths": [
                "/project/data/mouse/toliaslab/static/static20457-5-9-preproc0.zip"
            ],
            "normalize": True,
            "include_behavior": False,
            "include_eye_position": True,
            "batch_size": 40,
            "exclude": None,
            "file_tree": True,
            "scale": 1,
        },
        stored_paths=sorted(
            glob.glob("/project/presaved_models/pretrained_models/full/*")
        ),
        n_submodels=None,
    ):
        super().__init__()
        if n_submodels != None:
            stored_paths = stored_paths[-n_submodels:]
        dataloaders = get_data(dataset_fn, dataset_config)
        self.data_key = extract_data_key(dataset_config["paths"][0])
        self.dataloaders = dataloaders
        self.submodels_fn = submodels_fn
        self.submodels_config = submodels_config
        if type(stored_paths) != list:
            stored_paths = [stored_paths]
        self.submodels_paths = stored_paths
        self.submodels = [
            get_model(self.submodels_fn, self.submodels_config, dataloaders, seed=42)
            for _ in self.submodels_paths
        ]
        print("submodels used:")
        for i, path in enumerate(self.submodels_paths):
            print(path)
            self.submodels[i].load_state_dict(torch.load(path))
        self.submodels = nn.ModuleList(self.submodels)

    def forward(self, x):
        x = torch.stack([submodel(x) for submodel in self.submodels])
        x = torch.mean(x, dim=0)
        return x
