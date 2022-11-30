#%%
import matplotlib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class NormalizeClamp(nn.Module):
    def __init__(self, mean=0, std=0.1, dims=[1, 2, 3], min=-10, max=10):
        self.mean = mean
        self.std = std
        self.dims = dims
        self.min = min
        self.max = max

    def forward(self, x):
        shift = self.mean - x.mean(dims=self.dims, keepdim=True)
        x = x + shift
        gain = self.std / x.std(dims=self.dims, keepdim=True)
        x = gain * x
        x = torch.clamp(x, min=self.min, max=self.max)
        return x


def NormalizeClip(mean, std, pixel_min=None, pixel_max=None, dim=[1, 2, 3]):
    transf = [Normalize(mean=mean, std=std, dim=dim)]
    if pixel_min != None or pixel_max != None:
        transf.append(Clip(pixel_min=pixel_min, pixel_max=pixel_max))
    return nn.Sequential(*transf)


class Normalize(nn.Module):
    def __init__(self, mean=None, std=None, dim=None, eps=1e-12):
        super().__init__()
        self.mean = mean
        self.std = std
        self.dim = dim
        self.eps = eps

    def forward(self, x, iteration=None):
        x_mean = x.mean(dim=self.dim, keepdims=True)
        target_mean = self.mean if self.mean is not None else x_mean
        x_std = x.std(dim=self.dim, keepdims=True)
        target_std = self.std if self.std is not None else x_std
        return target_std * (x - x_mean) / (x_std + self.eps) + target_mean

    def __repr__(self):
        return f"Normalize(mean={self.mean}, std={self.std}, dim={self.dim})"


class Clip(nn.Module):
    def __init__(self, pixel_min, pixel_max):
        super().__init__()
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max

    def forward(self, x):
        x = torch.clamp(x, min=self.pixel_min, max=self.pixel_max)
        return x

    def __repr__(self):
        return f"Clip(min={self.pixel_min}, max={self.pixel_max})"
