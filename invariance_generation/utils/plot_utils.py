import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision.utils import make_grid


def plot_f(f, title="", vmin=None, vmax=None, return_plt=False, ticks=True, cmap=None):
    plt.clf()
    if type(f) == torch.Tensor:
        f = f.detach().cpu().numpy().squeeze()
    m = np.max(np.abs(f))
    if vmin is None:
        min = -m
    else:
        min = vmin
    if vmax is None:
        max = m
    else:
        max = vmax
    if cmap == "greys":
        color_map = "Greys_r"
    else:
        color_map = cm.coolwarm
    plt.imshow(f, vmax=max, vmin=min, cmap=color_map)
    plt.title(title)
    plt.colorbar()
    if ticks == False:
        plt.xticks([])
        plt.yticks([])
    if return_plt:
        return plt
    else:
        plt.show()


def plot_img(img, pixel_min, pixel_max):
    if type(img) != np.ndarray:
        img = img.cpu().detach().squeeze().numpy()
    plt.imshow(img, cmap="Greys_r", vmax=pixel_max, vmin=pixel_min)
    plt.colorbar()
    plt.show()


def plot_filters(filters, nrow, figsize=None, cmap=None, vmin=None, vmax=None):
    fig, ax = (
        plt.subplots(dpi=200)
        if figsize is None
        else plt.subplots(dpi=200, figsize=figsize)
    )
    image_grid = make_grid(filters, nrow=nrow).mean(0).cpu().data.numpy()
    if vmin == None:
        vmin = -np.abs(image_grid).max()
    if vmax == None:
        vmax = np.abs(image_grid).max()
    ax.imshow(image_grid, vmin=vmin, vmax=vmax, cmap=cmap)
    return fig, ax
