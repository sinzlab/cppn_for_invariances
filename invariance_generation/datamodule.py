import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# TODO decide how to deal with this
class JitteringGridDatamodule(nn.Module):
    def __init__(
        self,
        num_invariances,
        grid_points_per_dim,
        steps_per_epoch,
    ):
        super().__init__()
        self.num_invariances = num_invariances
        self.grid_points_per_dim = grid_points_per_dim
        self.steps_per_epoch = steps_per_epoch
        self.sigma = 2 * np.pi / self.grid_points_per_dim
        grid = torch.linspace(0, 2 * np.pi, self.grid_points_per_dim + 1)[:-1]
        self.grid = torch.stack(
            torch.meshgrid(*[grid for _ in range(self.num_invariances)]), -1
        ).flatten(0, -2)
        self.base_grid = self.grid.repeat(self.steps_per_epoch, 1)

    def train_dataloader(self):
        jitter = (
            torch.rand([self.steps_per_epoch, self.num_invariances]) * self.sigma
            - 0.5 * self.sigma
        )
        grids = self.base_grid + jitter.repeat_interleave(
            self.grid_points_per_dim**self.num_invariances, dim=0
        )
        return DataLoader(
            grids,
            batch_size=self.grid_points_per_dim**self.num_invariances,
        )
