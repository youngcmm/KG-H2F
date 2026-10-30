"""Composite scatter operations used by ULTRA's variadic utilities."""

from __future__ import annotations

import torch

from . import scatter_add, scatter_max


def scatter_softmax(src, index, dim=0, dim_size=None):
    maxima = scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
    expanded_max = maxima.index_select(dim, index)
    exp = torch.exp(src - expanded_max)
    denom = scatter_add(exp, index, dim=dim, dim_size=dim_size)
    return exp / denom.index_select(dim, index).clamp_min(torch.finfo(src.dtype).tiny)


def scatter_log_softmax(src, index, dim=0, dim_size=None):
    return scatter_softmax(src, index, dim=dim, dim_size=dim_size).clamp_min(
        torch.finfo(src.dtype).tiny
    ).log()

