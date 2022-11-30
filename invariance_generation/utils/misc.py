import numpy as np
import torch
import wandb


def prepare_video(img, pixel_min=None, pixel_max=None, fps=30):
    """Prepare video in Wandb format, so that it can be logged"""
    imgs_np = img.cpu().detach().numpy()
    imgs_np = np.tile(imgs_np, (1, 3, 1, 1))
    if pixel_min == None:
        pixel_min = -np.max(np.abs(imgs_np))
    if pixel_max == None:
        pixel_max = np.max(np.abs(imgs_np))
    imgs_video_in_scale = rescale(imgs_np, pixel_min, pixel_max, 0, 255)
    return wandb.Video(np.uint8(imgs_video_in_scale), fps=fps)


def flatten_list(t):
    return [item for sublist in t for item in sublist]


def rescale(x, in_min, in_max, out_min, out_max):
    in_mean = (in_min + in_max) / 2
    out_mean = (out_min + out_max) / 2
    in_ext = in_max - in_min
    out_ext = out_max - out_min
    gain = out_ext / in_ext
    x_rescaled = (x - in_mean) * gain + out_mean
    return x_rescaled


def standardize(x, dim=(1, 2, 3), return_shift_gain=False):
    shift = -x.mean(dim=dim, keepdim=True)
    gain = 1 / x.std(dim=dim, keepdim=True)
    x_standardized = (x + shift) * gain
    if return_shift_gain == True:
        return x_standardized, shift, gain
    else:
        return x_standardized


def normalize(x, dim=(1, 2, 3), return_shift_gain=False):
    shift = -x.mean(dim=dim, keepdim=True)
    x_shifted = x + shift
    gain = 1 / torch.linalg.norm(x_shifted, dim=dim, keepdim=True)
    x_normalized = x_shifted * gain
    if return_shift_gain == True:
        return x_normalized, shift, gain
    else:
        return x_normalized


def rescale_back(x, shift, gain):
    return (x / gain) - shift
