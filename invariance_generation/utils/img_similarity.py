# %%

import numpy as np
import torch
from scipy import spatial
import matplotlib.pyplot as plt


def cosine_similarity(tensor):
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    tensor_normed = tensor / torch.norm(tensor, p=2, dim=1, keepdim=True)
    return tensor_normed @ tensor_normed.T


def eucl_dist_similarity(x):
    # normalize
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / x.norm(dim=-1, keepdim=True)

    # make distance
    x = torch.cdist(x, x)
    x = -(x - 1)
    return x


def dot_product_similarity(x):
    x_normed = x / x.norm(dim=1, p=2, keepdim=True)
    return x_normed @ x_normed.T


# ### test example
# n_cols = 5
# dd = np.random.randn(100, n_cols).astype(np.float32)

# # using scipy
# scipy_results = np.array(
#     [
#         1 - spatial.distance.cosine(dd.T[i], dd.T[j])
#         for i in range(n_cols)
#         for j in range(n_cols)
#     ]
# ).reshape(n_cols, n_cols)

# # using torch
# torch_results = cosine_similarity(torch.from_numpy(dd).T)

# # check if the results are close enough
# assert (
#     (torch.from_numpy(scipy_results.astype(np.float32)) - torch_results) < 1e-5
# ).all()


# # visualize the results
# fix, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2), dpi=200)
# ax1.imshow(scipy_results)
# ax1.set_title("Scipy")
# for i in range(n_cols):
#     for j in range(n_cols):
#         text = ax1.text(
#             j,
#             i,
#             f"{scipy_results[i, j]:.2f}",
#             ha="center",
#             va="center",
#             color="w",
#             fontsize=7,
#         )

# ax2.imshow(torch_results)
# ax2.set_title("Torch")
# for i in range(n_cols):
#     for j in range(n_cols):
#         text = ax2.text(
#             j,
#             i,
#             f"{torch_results[i, j]:.2f}",
#             ha="center",
#             va="center",
#             color="w",
#             fontsize=7,
#         )

# # %%

# # %%
