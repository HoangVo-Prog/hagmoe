from __future__ import annotations

import torch
import torch.nn as nn


def gated_fusion_two(gate: nn.Module, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    g = torch.sigmoid(gate(torch.cat([a, b], dim=-1)))
    return g * a + (1 - g) * b


def gated_fusion_three(
    gate: nn.Module,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    b_scale: float = 1.0,
    c_scale: float = 1.0,
) -> torch.Tensor:
    g = torch.sigmoid(gate(torch.cat([a, b, c], dim=-1)))
    return g * a + (1 - g) * (b_scale * b + c_scale * c)


def topk_renorm(probs: torch.Tensor, k: int) -> torch.Tensor:
    """
    Top-k soft routing: keep k largest probs per row, zero others, renormalize.

    Example (k=2):
        probs = [0.6, 0.3, 0.1] -> mask -> [0.6, 0.3, 0.0]
        renorm by sum=0.9 -> [0.6667, 0.3333, 0.0]
    """
    if k <= 0 or k >= probs.size(-1):
        return probs
    _, topk_idx = torch.topk(probs, k=k, dim=-1)
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(1, topk_idx, True)
    masked = torch.where(mask, probs, torch.zeros_like(probs))
    denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return masked / denom


def topk_soft_routing(
    probs: torch.Tensor,
    *,
    top_k: int,
    normalization: str = "renorm",
    logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Enforce soft top-k routing by masking all but top_k entries and renormalizing.
    """
    if top_k <= 0 or top_k >= probs.size(-1):
        return probs
    _, topk_idx = torch.topk(probs, k=top_k, dim=-1)
    if normalization == "softmax" and logits is not None:
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(1, topk_idx, logits.gather(1, topk_idx))
        return torch.softmax(masked, dim=-1)

    return topk_renorm(probs, top_k)
