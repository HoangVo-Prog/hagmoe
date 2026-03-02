from __future__ import annotations

import torch


def masked_mean(
    x: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    dim: int,
    keepdim: bool = False,
) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim, keepdim=keepdim)
    mask_f = mask.float()
    denom = mask_f.sum(dim=dim, keepdim=True).clamp_min(1.0)
    if x.dim() == mask_f.dim():
        weighted = x * mask_f
        out = weighted.sum(dim=dim, keepdim=True) / denom
    else:
        weighted = x * mask_f.unsqueeze(-1)
        out = weighted.sum(dim=dim, keepdim=True) / denom.unsqueeze(-1)
    if not keepdim:
        out = out.squeeze(dim)
    return out
