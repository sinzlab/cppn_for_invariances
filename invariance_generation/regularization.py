#%%
import itertools
import torch
import torch.nn as nn
import numpy as np
from invariance_generation.utils.img_similarity import (
    cosine_similarity,
    eucl_dist_similarity,
    dot_product_similarity,
)


def l1(x, gamma=0.1):
    return gamma * x.pow(2).mean()


def l2(x, gamma=0.1):
    return gamma * torch.abs(x).mean()


def GetPosNegMask(
    num_invariances,
    grid_points_per_dim,
    neighbor_size,
    with_periodic_invariances,
    with_round_neighbor,
):
    masks = np.ones([grid_points_per_dim] * 2 * num_invariances)
    masks = np.ones([grid_points_per_dim] * 2 * num_invariances)
    for point in itertools.product(
        np.arange(grid_points_per_dim), repeat=num_invariances
    ):
        masks[tuple(point)] = get_mask_of_point(
            point,
            grid_points_per_dim,
            num_invariances,
            neighbor_size,
            with_periodic_invariances,
            with_round_neighbor,
        )
    return masks


def get_mask_of_point(
    point,
    grid_points_per_dim,
    num_invariances,
    neighbor_size,
    with_periodic_invariances,
    with_round_neighbor=False,
):
    assert num_invariances == 1 or num_invariances == 2
    ns = int(neighbor_size * grid_points_per_dim)
    if num_invariances == 1:
        assert type(with_periodic_invariances) == bool
        mask = -torch.ones([grid_points_per_dim] * num_invariances)
        x = point[0]
        if x < ns:
            mask[: x + ns + 1] = 1
            if with_periodic_invariances == True:
                mask[x - ns :] = 1
        elif x + ns >= grid_points_per_dim:
            mask[x - ns :] = 1
            if with_periodic_invariances == True:
                mask[: ns - grid_points_per_dim + x + 1] = 1
        else:
            mask[x - ns : x + ns + 1] = 1
        mask[x] = 0
        return mask
    if num_invariances == 2:
        point = list(point)
        if type(with_periodic_invariances) == bool:
            with_periodic_invariances = [with_periodic_invariances] * 2
        if type(with_periodic_invariances) == list:
            assert len(with_periodic_invariances) == 2
        mask_n_points = grid_points_per_dim + 2 * ns
        mask_size = [grid_points_per_dim] * 2
        for dim, periodicity in enumerate(with_periodic_invariances):
            if periodicity == False:
                mask_size[dim] = mask_n_points
                point[dim] = point[dim] + ns
        mask = -torch.ones(mask_size)
        start = [int(s / 2) for s in mask_size]
        mask[start[0] - ns : start[0] + ns + 1, start[1] - ns : start[1] + ns + 1] = 1
        mask[start[0], start[1]] = 0
        if with_round_neighbor:
            pos_idxs = np.argwhere(mask > 0)
            pos_idxs = np.array(pos_idxs).T

            for idx in pos_idxs:
                idx = tuple(idx)
                dist_x = idx[0] - start[0]
                dist_y = idx[1] - start[1]
                dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
                if dist > ns:
                    mask[idx] = -1
        translation = [p - s for p, s in zip(point, start)]

        mask = np.roll(mask, shift=tuple(translation), axis=(0, 1))

        if with_periodic_invariances[0] == False:
            mask = mask[ns : ns + grid_points_per_dim, :]
        if with_periodic_invariances[1] == False:
            mask = mask[:, ns : ns + grid_points_per_dim]
        return mask


# from invariance_generation.utils.plot_utils import plot_f

# plot_f(get_mask_of_point([0, 0], 30, 2, 0.2, True, True))

#%%
#%%


class SimCLROnGrid(nn.Module):
    def __init__(
        self,
        num_invariances,
        grid_points_per_dim,
        neighbor_size,
        temperature,
        with_periodic_invariances,
        with_round_neighbor=False,
        **args
    ):
        super().__init__()
        self.num_invariances = num_invariances
        self.points_per_dim = grid_points_per_dim
        self.neighbor_size = neighbor_size
        self.temperature = temperature
        self.with_round_neighbor = with_round_neighbor
        self.neighbor_mask = torch.tensor(
            GetPosNegMask(
                num_invariances,
                grid_points_per_dim,
                neighbor_size,
                with_periodic_invariances,
                with_round_neighbor,
            )
        )
        self.register_buffer(
            "flat_neighbor_mask",
            self.neighbor_mask.reshape(
                self.points_per_dim ** self.num_invariances,
                self.points_per_dim ** self.num_invariances,
            ),
        )
        self.register_buffer(
            "flat_neighbor_mask_pos",
            torch.where(self.flat_neighbor_mask > 0, self.flat_neighbor_mask, 0.0),
        )
        self.register_buffer(
            "flat_neighbor_mask_neg",
            torch.where(
                self.flat_neighbor_mask < 0,
                self.flat_neighbor_mask,
                0.0,
            ),
        )

    def reg_term(self, images):
        images = images.flatten(
            1, -1
        )  # [N, c, h, w] -> [N, c * h * w]  N=number of points in the grid
        similarity = torch.exp(cosine_similarity(images) / self.temperature)  # [N, N]
        num = (similarity * self.flat_neighbor_mask_pos).sum(
            dim=-1
        ) / self.flat_neighbor_mask_pos.sum(
            dim=-1
        )  # [N]
        den = -(-similarity * self.flat_neighbor_mask_neg).sum(
            dim=-1
        ) / self.flat_neighbor_mask_neg.sum(
            dim=-1
        )  # [N]
        reg_term = torch.log(num / den).mean()
        return reg_term * self.temperature / 2

    def pos_term(self, images):
        images = images.flatten(1, -1)
        similarity = torch.exp(cosine_similarity(images) / self.temperature)
        num = (similarity * self.flat_neighbor_mask_pos).sum(
            dim=-1
        ) / self.flat_neighbor_mask_pos.sum(dim=-1)
        return torch.log(num).mean() * self.temperature / 2

    def neg_term(self, images):
        images = images.flatten(1, -1)
        similarity = torch.exp(cosine_similarity(images) / self.temperature)
        den = -(-similarity * self.flat_neighbor_mask_neg).sum(
            dim=-1
        ) / self.flat_neighbor_mask_neg.sum(dim=-1)
        return -torch.log(den).mean() * self.temperature / 2


def GetSimCLROnGrid(
    num_invariances,
    grid_points_per_dim,
    neighbor_size,
    temperature,
    with_periodic_invariances,
    **args
):
    if with_periodic_invariances == True:
        reg = TemperatureContrastiveRegularizationOnPeriodicGrid(
            num_invariances=num_invariances,
            grid_points_per_dim=grid_points_per_dim,
            neighbor_size=neighbor_size,
            temperature=temperature,
        )
    if with_periodic_invariances == False:
        reg = TemperatureContrastiveRegularizationOnNonPeriodicGrid(
            num_invariances=num_invariances,
            grid_points_per_dim=grid_points_per_dim,
            neighbor_size=neighbor_size,
            temperature=temperature,
        )
    return reg


class TemperatureContrastiveRegularizationOnPeriodicGrid(nn.Module):
    def __init__(
        self, num_invariances, grid_points_per_dim, neighbor_size, temperature, **args
    ):
        super().__init__()
        self.num_invariances = num_invariances
        self.points_per_dim = grid_points_per_dim
        self.neighbor_size = neighbor_size
        self.temperature = temperature
        self.neighbor_mask = np.ones([self.points_per_dim] * 2 * self.num_invariances)
        for point in itertools.product(
            np.arange(self.points_per_dim), repeat=self.num_invariances
        ):
            self.neighbor_mask[tuple(point)] = self.get_mask_of_point(
                point, self.points_per_dim, self.num_invariances
            )
        self.neighbor_mask = torch.tensor(self.neighbor_mask)
        self.register_buffer(
            "flat_neighbor_mask",
            self.neighbor_mask.reshape(
                self.points_per_dim ** self.num_invariances,
                self.points_per_dim ** self.num_invariances,
            ),
        )
        self.register_buffer(
            "flat_neighbor_mask_pos",
            torch.where(self.flat_neighbor_mask > 0, self.flat_neighbor_mask, 0.0),
        )
        self.register_buffer(
            "flat_neighbor_mask_neg",
            torch.where(
                self.flat_neighbor_mask < 0,
                self.flat_neighbor_mask,
                0.0,
            ),
        )

    def find_neighbours(self, point_idx, points_per_dim, dim):
        neighbors_list = []
        neigh_size = int(points_per_dim * self.neighbor_size)
        shifts = list(
            itertools.product(np.arange(-neigh_size, neigh_size + 1), repeat=dim)
        )
        shifts.remove(tuple([0] * self.num_invariances))
        for shift in shifts:
            ne = np.remainder(np.array(point_idx) + np.array(shift), points_per_dim)
            neighbors_list.append(ne)
        return neighbors_list

    def get_mask_of_point(self, point, points_per_dim, dim):
        mask = -torch.ones([points_per_dim] * dim)
        mask[tuple(point)] = 0
        neighbors_list = self.find_neighbours(point, points_per_dim, dim)
        for ne in neighbors_list:
            mask[tuple(ne)] = 1
        return mask

    def reg_term(self, images):
        images = images.flatten(
            1, -1
        )  # [N, c, h, w] -> [N, c * h * w]  N=number of points in the grid
        similarity = torch.exp(cosine_similarity(images) / self.temperature)  # [N, N]
        num = (
            similarity[self.flat_neighbor_mask > 0]
            .reshape(self.num_invariances * self.points_per_dim, -1)
            .mean(dim=-1)
        )  # [N]
        den = (
            similarity[self.flat_neighbor_mask < 0]
            .reshape(self.num_invariances * self.points_per_dim, -1)
            .mean(dim=-1)
        )  # [N]
        reg_term = torch.log(num / den).mean()
        return reg_term

    def pos_term(self, images):
        images = images.flatten(1, -1)
        similarity = torch.exp(cosine_similarity(images) / self.temperature)
        num = (
            similarity[self.flat_neighbor_mask > 0]
            .reshape(self.num_invariances * self.points_per_dim, -1)
            .mean(dim=-1)
        )
        return torch.log(num).mean()

    def neg_term(self, images):
        images = images.flatten(1, -1)
        similarity = torch.exp(cosine_similarity(images) / self.temperature)
        den = (
            similarity[self.flat_neighbor_mask < 0]
            .reshape(self.num_invariances * self.points_per_dim, -1)
            .mean(dim=-1)
        )
        return -torch.log(den).mean()


class TemperatureContrastiveRegularizationOnNonPeriodicGrid(nn.Module):
    def __init__(
        self, num_invariances, grid_points_per_dim, neighbor_size, temperature, **args
    ):
        super().__init__()
        self.num_invariances = num_invariances
        self.points_per_dim = grid_points_per_dim
        self.neighbor_size = neighbor_size
        self.temperature = temperature
        # get all mask of all points
        self.neighbor_mask = np.ones([self.points_per_dim] * 2 * self.num_invariances)
        for point in itertools.product(
            np.arange(self.points_per_dim), repeat=self.num_invariances
        ):
            self.neighbor_mask[tuple(point)] = self.get_mask_of_point(point)
        self.neighbor_mask = torch.tensor(self.neighbor_mask)
        # register buffers
        self.register_buffer(
            "flat_neighbor_mask",
            self.neighbor_mask.reshape(
                self.points_per_dim ** self.num_invariances,
                self.points_per_dim ** self.num_invariances,
            ),
        )
        self.register_buffer(
            "flat_neighbor_mask_pos",
            torch.where(self.flat_neighbor_mask > 0, self.flat_neighbor_mask, 0.0),
        )
        self.register_buffer(
            "flat_neighbor_mask_neg",
            torch.where(
                self.flat_neighbor_mask < 0,
                self.flat_neighbor_mask,
                0.0,
            ),
        )

    def get_mask_of_point(self, point):
        ns = int(self.neighbor_size * self.points_per_dim)
        if self.num_invariances == 1:
            mask = -torch.ones([self.points_per_dim] * self.num_invariances)
            x = point[0]
            if x <= ns:
                mask[: x + ns + 1] = 1
            elif x + ns >= self.points_per_dim:
                mask[x - ns :] = 1
            else:
                mask[x - ns : x + ns + 1] = 1
        if self.num_invariances == 2:
            big_mask = -torch.ones(
                [self.points_per_dim + 2 * ns] * self.num_invariances
            )
            i, j = point
            big_mask[i : i + 2 * ns + 1, j : j + 2 * ns + 1] = 1
            mask = big_mask[
                ns : self.points_per_dim + ns, ns : self.points_per_dim + ns
            ]
        mask[point] = 0
        return mask

    def reg_term(self, images):
        images = images.flatten(
            1, -1
        )  # [N, c, h, w] -> [N, c * h * w]  N=number of points in the grid
        similarity = torch.exp(cosine_similarity(images) / self.temperature)  # [N, N]
        num = (similarity * self.flat_neighbor_mask_pos).sum(
            dim=-1
        ) / self.flat_neighbor_mask_pos.sum(
            dim=-1
        )  # [N]
        den = -(-similarity * self.flat_neighbor_mask_neg).sum(
            dim=-1
        ) / self.flat_neighbor_mask_neg.sum(
            dim=-1
        )  # [N]
        reg_term = torch.log(num / den).mean()
        return reg_term

    def pos_term(self, images):
        images = images.flatten(1, -1)
        similarity = torch.exp(cosine_similarity(images) / self.temperature)
        num = (similarity * self.flat_neighbor_mask_pos).sum(
            dim=-1
        ) / self.flat_neighbor_mask_pos.sum(dim=-1)
        return torch.log(num).mean()

    def neg_term(self, images):
        images = images.flatten(1, -1)
        similarity = torch.exp(cosine_similarity(images) / self.temperature)
        den = -(-similarity * self.flat_neighbor_mask_neg).sum(
            dim=-1
        ) / self.flat_neighbor_mask_neg.sum(dim=-1)
        return -torch.log(den).mean()


class PeriodicGridRegularization(nn.Module):
    def __init__(
        self,
        num_invariances,
        grid_points_per_dim,
        neighbor_size,
        gamma_pos,
        gamma_neg,
        **args
    ):
        super().__init__()
        self.num_invariances = num_invariances
        self.points_per_dim = grid_points_per_dim
        self.neighbor_size = neighbor_size
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.neighbor_mask = np.ones([self.points_per_dim] * 2 * num_invariances)
        self.neighbor_mask = np.ones([self.points_per_dim] * 2 * self.num_invariances)
        for point in itertools.product(
            np.arange(self.points_per_dim), repeat=self.num_invariances
        ):
            self.neighbor_mask[tuple(point)] = self.get_mask_of_point(
                point, self.points_per_dim, self.num_invariances
            )
        self.neighbor_mask = torch.tensor(self.neighbor_mask)
        me = self.neighbor_mask[[0] * self.num_invariances]
        self.neighbor_mask[self.neighbor_mask > 0] *= self.gamma_pos / torch.abs(
            torch.sum(me[me > 0])
        )
        self.neighbor_mask[self.neighbor_mask < 0] *= self.gamma_neg / torch.abs(
            torch.sum(me[me < 0])
        )
        self.register_buffer(
            "flat_neighbor_mask",
            self.neighbor_mask.reshape(
                self.points_per_dim ** self.num_invariances,
                self.points_per_dim ** self.num_invariances,
            ),
        )
        self.register_buffer(
            "flat_neighbor_mask_pos",
            torch.where(self.flat_neighbor_mask > 0, self.flat_neighbor_mask, 0.0),
        )
        self.register_buffer(
            "flat_neighbor_mask_neg",
            torch.where(
                self.flat_neighbor_mask < 0,
                self.flat_neighbor_mask,
                0.0,
            ),
        )

    def find_neighbours(self, point_idx, points_per_dim, dim):
        neighbors_list = []
        neigh_size = int(points_per_dim * self.neighbor_size)
        shifts = list(
            itertools.product(np.arange(-neigh_size, neigh_size + 1), repeat=dim)
        )
        shifts.remove(tuple([0] * self.num_invariances))
        for shift in shifts:
            ne = np.remainder(np.array(point_idx) + np.array(shift), points_per_dim)
            neighbors_list.append(ne)
        return neighbors_list

    def get_mask_of_point(self, point, points_per_dim, dim):
        mask = -torch.ones([points_per_dim] * dim)
        mask[tuple(point)] = 0
        neighbors_list = self.find_neighbours(point, points_per_dim, dim)
        for ne in neighbors_list:
            mask[tuple(ne)] = 1
        return mask

    def similarity_nonlin_transformation(self, x):
        # this can be modified with other nonlin_transfs
        x = 2 * torch.nn.functional.softplus(5 * x) / 5 - 1.0027
        return x

    def reg_term(self, images):
        images = images.flatten(1, -1)
        similarity = self.similarity_nonlin_transformation(
            eucl_dist_similarity(images) + 1
        )
        reg_loss = torch.sum(similarity * self.flat_neighbor_mask) / len(images)
        return reg_loss

    def pos_term(self, images):
        images = images.flatten(1, -1)
        similarity = self.similarity_nonlin_transformation(
            eucl_dist_similarity(images) + 1
        )
        reg_loss = torch.sum(similarity * self.flat_neighbor_mask_pos) / len(images)
        return reg_loss

    def neg_term(self, images):
        images = images.flatten(1, -1)
        similarity = self.similarity_nonlin_transformation(
            eucl_dist_similarity(images) + 1
        )
        reg_loss = torch.sum(similarity * self.flat_neighbor_mask_neg) / len(images)
        return reg_loss


class GridRegularization(nn.Module):
    def __init__(
        self,
        num_invariances,
        grid_points_per_dim,
        neighbor_size,
        gamma_pos,
        gamma_neg,
        absolute_similarity=True,
        with_periodic_invariances=True,
        **args
    ):
        super().__init__()
        self.num_invariances = num_invariances
        self.points_per_dim = grid_points_per_dim
        self.neighbor_size = neighbor_size
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.absolute_similarity = absolute_similarity
        self.neighbor_mask = np.ones([self.points_per_dim] * 2 * num_invariances)
        self.neighbor_mask = np.ones([self.points_per_dim] * 2 * self.num_invariances)
        for point in itertools.product(
            np.arange(self.points_per_dim), repeat=self.num_invariances
        ):
            self.neighbor_mask[tuple(point)] = self.get_mask_of_point(
                point, self.points_per_dim, self.num_invariances
            )
        self.neighbor_mask = torch.tensor(
            GetPosNegMask(
                num_invariances,
                grid_points_per_dim,
                neighbor_size,
                with_periodic_invariances,
            )
        )
        self.register_buffer(
            "flat_neighbor_mask",
            self.neighbor_mask.reshape(
                self.points_per_dim ** self.num_invariances,
                self.points_per_dim ** self.num_invariances,
            ),
        )
        self.register_buffer(
            "flat_neighbor_mask_pos",
            torch.where(self.flat_neighbor_mask > 0, 1.0, 0.0),
        )
        self.register_buffer(
            "flat_neighbor_mask_neg",
            torch.where(
                self.flat_neighbor_mask < 0,
                1.0,
                0.0,
            ),
        )

    def find_neighbours(self, point_idx, points_per_dim, dim):
        neighbors_list = []
        neigh_size = int(points_per_dim * self.neighbor_size)
        shifts = list(
            itertools.product(np.arange(-neigh_size, neigh_size + 1), repeat=dim)
        )
        shifts.remove(tuple([0] * self.num_invariances))
        for shift in shifts:
            ne = np.remainder(np.array(point_idx) + np.array(shift), points_per_dim)
            neighbors_list.append(ne)
        return neighbors_list

    def get_mask_of_point(self, point, points_per_dim, dim):
        mask = -torch.ones([points_per_dim] * dim)
        mask[tuple(point)] = 0
        neighbors_list = self.find_neighbours(point, points_per_dim, dim)
        for ne in neighbors_list:
            mask[tuple(ne)] = 1
        return mask

    def similarity(self, images):
        images = images.flatten(1, -1)
        return dot_product_similarity(images)

    def pos_term(self, images):
        similarity_transform = torch.abs if self.absolute_similarity else lambda x: x
        similarity = similarity_transform(self.similarity(images))
        similarity_masked = similarity * self.flat_neighbor_mask_pos
        reg_loss = similarity_masked.sum() / self.flat_neighbor_mask_pos.sum()
        return self.gamma_pos * reg_loss

    def neg_term(self, images):
        similarity_transform = torch.abs if self.absolute_similarity else lambda x: x
        similarity = similarity_transform(self.similarity(images))
        similarity_masked = similarity * self.flat_neighbor_mask_neg
        reg_loss = similarity_masked.sum() / self.flat_neighbor_mask_neg.sum()
        return self.gamma_neg * reg_loss

    def reg_term(self, images):
        return self.pos_term(images) - self.neg_term(images)
