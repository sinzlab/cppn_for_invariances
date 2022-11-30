import torch
from classicalv1.toy_models import *
from invariance_generation.models import *


def load_stored(model_class, path, map_location=None):
    hparams_and_state_dict = torch.load(path, map_location)
    model = eval(model_class)(**hparams_and_state_dict["hparams"])
    model.load_state_dict(hparams_and_state_dict["model_state_dict"])
    return model
