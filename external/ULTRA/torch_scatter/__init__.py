"""Native-PyTorch compatibility subset of :mod:`torch_scatter`.

The Windows Python 3.13 / PyTorch 2.12 environment has no matching binary
wheel for torch-scatter.  ULTRA only needs the reductions implemented here;
no learned parameters or model equations are changed.
"""

from __future__ import annotations

import torch


def _expanded_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim += src.dim()
    if index.dim() == 1:
        shape = [1] * src.dim()
        shape[dim] = index.numel()
        index = index.reshape(shape)
    return index.expand_as(src)


def _output_shape(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size):
    if dim < 0:
        dim += src.dim()
    shape = list(src.shape)
    inferred = int(index.max().item()) + 1 if index.numel() else 0
    shape[dim] = int(dim_size) if dim_size is not None else inferred
    return shape


def scatter_add(src, index, dim=0, out=None, dim_size=None):
    expanded = _expanded_index(index, src, dim)
    if out is None:
        out = torch.zeros(
            _output_shape(src, expanded, dim, dim_size),
            dtype=src.dtype,
            device=src.device,
        )
    return out.scatter_add_(dim, expanded, src)


def scatter_mean(src, index, dim=0, out=None, dim_size=None):
    result = scatter_add(src, index, dim=dim, out=out, dim_size=dim_size)
    count = scatter_add(
        torch.ones_like(src), index, dim=dim, dim_size=result.shape[dim]
    ).clamp_min_(1)
    return result / count


def scatter_max(src, index, dim=0, out=None, dim_size=None):
    expanded = _expanded_index(index, src, dim)
    if out is None:
        out = torch.full(
            _output_shape(src, expanded, dim, dim_size),
            -torch.inf,
            dtype=src.dtype,
            device=src.device,
        )
    values = out.scatter_reduce_(
        dim, expanded, src, reduce="amax", include_self=True
    )
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
        expanded = _expanded_index(index, src, dim)
        if out is None:
            out = torch.full(
                _output_shape(src, expanded, dim, dim_size),
                torch.inf,
                dtype=src.dtype,
                device=src.device,
            )
        values = out.scatter_reduce_(
            dim, expanded, src, reduce="amin", include_self=True
        )
        return torch.where(torch.isinf(values), torch.zeros_like(values), values)
    raise ValueError(f"Unsupported reduction: {reduce}")

