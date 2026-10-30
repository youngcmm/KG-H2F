"""Small CPU-compatible subset of torch_scatter used by KG-ICL.

The official extension has no wheel for the workspace's Python/PyTorch pair.
These implementations use native PyTorch scatter operations and preserve the
interfaces exercised by KG-ICL inference.
"""

from __future__ import annotations

import torch


def _expanded_index(index, src, dim):
    if dim < 0:
        dim += src.dim()
    shape = [1] * src.dim()
    shape[dim] = index.numel()
    return index.reshape(shape).expand_as(src)


def _shape(src, dim, dim_size):
    shape = list(src.shape)
    shape[dim] = int(dim_size if dim_size is not None else 1)
    return shape


def scatter_add(src, index, dim=0, out=None, dim_size=None):
    if out is None:
        out = torch.zeros(_shape(src, dim, dim_size), dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, _expanded_index(index, src, dim), src)


def scatter_mean(src, index, dim=0, out=None, dim_size=None):
    result = scatter_add(src, index, dim=dim, out=out, dim_size=dim_size)
    ones = torch.ones_like(src)
    count = scatter_add(ones, index, dim=dim, dim_size=result.shape[dim]).clamp_min_(1)
    return result / count


def scatter_max(src, index, dim=0, out=None, dim_size=None):
    if out is None:
        out = torch.full(_shape(src, dim, dim_size), -torch.inf, dtype=src.dtype, device=src.device)
    values = out.scatter_reduce_(dim, _expanded_index(index, src, dim), src, reduce="amax", include_self=True)
    values = torch.where(torch.isinf(values), torch.zeros_like(values), values)
    return values, torch.full_like(values, -1, dtype=torch.long)


def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    if reduce in {"sum", "add"}:
        return scatter_add(src, index, dim=dim, out=out, dim_size=dim_size)
    if reduce == "mean":
        return scatter_mean(src, index, dim=dim, out=out, dim_size=dim_size)
    if reduce == "max":
        return scatter_max(src, index, dim=dim, out=out, dim_size=dim_size)[0]
    if reduce == "min":
        if out is None:
            out = torch.full(_shape(src, dim, dim_size), torch.inf, dtype=src.dtype, device=src.device)
        values = out.scatter_reduce_(dim, _expanded_index(index, src, dim), src, reduce="amin", include_self=True)
        return torch.where(torch.isinf(values), torch.zeros_like(values), values)
    raise ValueError(f"Unsupported reduction: {reduce}")
