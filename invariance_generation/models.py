import torch
import torch.nn as nn
from nnfabrik.builder import get_data, get_model
from invariance_generation.utils import extract_data_key
import glob
from invariance_generation.utils.model_utils import se_core_point_readout


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


data_info = {
    "3631896544452": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 32,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3632669014376": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 21,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3632932714885": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 11,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3633364677437": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 10,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3634055946316": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 21,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3634142311627": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 14,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3634658447291": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 5,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3634744023164": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 12,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3635178040531": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 6,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3635949043110": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 10,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3636034866307": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 20,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3636552742293": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 24,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3637161140869": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 22,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3637248451650": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 7,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3637333931598": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 9,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3637760318484": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 20,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3637851724731": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 14,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3638367026975": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 16,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3638456653849": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 2,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3638885582960": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 5,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3638373332053": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 10,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3638541006102": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 17,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3638802601378": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 7,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3638973674012": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 18,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3639060843972": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 12,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3639406161189": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 14,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3640011636703": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 2,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3639664527524": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 18,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3639492658943": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 17,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3639749909659": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 8,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3640095265572": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 25,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3631807112901": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 29,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
}

model_config = {
    "pad_input": False,
    "stack": -1,
    "depth_separable": True,
    "input_kern": 24,
    "gamma_input": 10,
    "gamma_readout": 0.5,
    "hidden_dilation": 2,
    "hidden_kern": 9,
    "se_reduction": 16,
    "n_se_blocks": 2,
    "hidden_channels": 32,
    "linear": False,
}


def get_model(seed, data_info=data_info, model_config=model_config):
    model = se_core_point_readout(
        dataloaders=None, seed=seed, data_info=data_info, **model_config
    )
    return model
