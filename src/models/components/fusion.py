from __future__ import annotations

import torch
import torch.nn as nn


def bilinear_fusion(
    proj_a: nn.Module,
    proj_b: nn.Module,
    out: nn.Module,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return out(proj_a(a) * proj_b(b))


def bilinear_fusion_three(
    proj_a: nn.Module,
    proj_b: nn.Module,
    out: nn.Module,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    return out(proj_a(a) * proj_b(b) * proj_b(c))
