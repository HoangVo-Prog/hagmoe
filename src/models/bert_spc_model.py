from __future__ import annotations

import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.core.loss.focal_loss import FocalLoss
from src.models.components.heads import build_head


class BERTSPCModel(nn.Module):
    def __init__(
        self,
        *,
        encoder=None,
        model_cfg: dict,
        loss_cfg: dict,
        moe_cfg: dict | None = None,
    ) -> None:
        super().__init__()

        if encoder is None:
            encoder_name = model_cfg["encoder"]["pretrained_name"]
            self.encoder = AutoModel.from_pretrained(encoder_name)
        elif isinstance(encoder, str):
            self.encoder = AutoModel.from_pretrained(encoder)
        else:
            self.encoder = encoder

        hidden_size = int(getattr(self.encoder.config, "hidden_size"))
        num_labels = int(model_cfg["common"]["num_labels"])
        fusion_cfg = model_cfg.get("fusion", {})
        head_cfg = model_cfg.get("head", {})
        dropout = float(fusion_cfg.get("dropout", model_cfg["encoder"].get("dropout", 0.1)))
        head_type = str(head_cfg.get("type", "linear"))

        self.dropout = nn.Dropout(dropout)
        self.head = build_head(head_type, hidden_size, num_labels, dropout)

        self.loss_type = str(loss_cfg["type"]).lower().strip()
        self.focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))

        cw = None
        if class_weights is None:
            cw = None
        elif isinstance(class_weights, torch.Tensor):
            cw = class_weights.detach().float()
        else:
            s = str(class_weights).strip()
            if not s:
                cw = None
            elif s.startswith("[") and s.endswith("]"):
                vals = ast.literal_eval(s)
                cw = torch.tensor([float(v) for v in vals], dtype=torch.float)
            else:
                cw = torch.tensor(
                    [float(x.strip()) for x in s.split(",") if x.strip()],
                    dtype=torch.float,
                )

        self.register_buffer("class_weights", cw)

        if self.loss_type not in {"ce", "weighted_ce", "focal"}:
            raise ValueError("loss_type must be one of: ce, weighted_ce, focal")

        if self.loss_type in {"weighted_ce", "focal"} and self.class_weights is None:
            raise ValueError("class_weights must be provided for weighted_ce or focal")

    def forward(
        self,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        labels=None,
        fusion_method: str = "concat",
    ):
        outputs = self.encoder(
            input_ids=input_ids_sent,
            attention_mask=attention_mask_sent,
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.head(self.dropout(cls_repr))
        return self._compute_loss(logits, labels)

    def _compute_loss(self, logits, labels):
        if labels is None:
            return {"loss": None, "logits": logits}

        if self.loss_type == "ce":
            loss = F.cross_entropy(logits, labels)
        elif self.loss_type == "weighted_ce":
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss = F.cross_entropy(logits, labels, weight=w)
        elif self.loss_type == "focal":
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss_fn = FocalLoss(gamma=self.focal_gamma, alpha=w, reduction="mean")
            loss = loss_fn(logits, labels)
        else:
            raise RuntimeError(f"Unexpected loss_type: {self.loss_type}")

        return {"loss": loss, "logits": logits}
