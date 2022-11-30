import torch


def nonlinearity_idx(nl_corr, l_corr):
    return (1 - l_corr) * nl_corr


def select_nonlinear_neurons(nl_corr, l_corr, nl_threshold, nl_idx_threshold):
    nl_idx = nonlinearity_idx(nl_corr, l_corr)
    nl_cond = nl_corr > nl_threshold
    nl_idx_cond = nl_idx > nl_idx_threshold
    idx_passing = nl_idx_cond * nl_cond * nl_idx
    return idx_passing


def nonlinearity_idx2(nl_corr, l_corr):
    return (nl_corr - torch.max(l_corr, torch.zeros_like(l_corr))) / nl_corr
