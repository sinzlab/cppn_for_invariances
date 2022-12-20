import numpy as np
import torch
from invariance_generation.utils.model_utils import se_core_point_readout


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


def get_model(seed=None, data_info=data_info, model_config=model_config):
    seed = seed if seed is not None else np.random.randint(0, 100)
    model = se_core_point_readout(
        dataloaders=None, seed=seed, data_info=data_info, **model_config
    )
    return model
