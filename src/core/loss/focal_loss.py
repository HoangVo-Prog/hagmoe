import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    Args:
        gamma: focusing parameter
        alpha: optional class weights (same semantics as CrossEntropy weight)
        reduction: "mean" | "sum" | "none"
    """
    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha= None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none")
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE per sample
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
        )  # [B]

        # pt = P(correct class)
        pt = torch.exp(-ce).clamp_min(1e-8)  # [B]
        loss = ((1.0 - pt) ** self.gamma) * ce  # [B]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
