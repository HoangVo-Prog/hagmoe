from __future__ import annotations

import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.core.loss.focal_loss import FocalLoss
from src.models.components.heads import build_head
from src.models.components.gating import gated_fusion_two
from src.models.components.fusion import bilinear_fusion
from src.models.components.pooling import masked_mean


class BaseModel(nn.Module):
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

        _candidates = [8, 4, 2, 1]
        num_heads = next((x for x in _candidates if hidden_size % x == 0), 1)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.coattn_term_to_sent = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.coattn_sent_to_term = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.gate = nn.Linear(2 * hidden_size, hidden_size)

        bilinear_rank = max(32, min(256, hidden_size // 4))
        self.bilinear_proj_sent = nn.Linear(hidden_size, bilinear_rank)
        self.bilinear_proj_term = nn.Linear(hidden_size, bilinear_rank)
        self.bilinear_out = nn.Linear(bilinear_rank, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.head_single = build_head(head_type, hidden_size, num_labels, dropout)
        self.head_concat = build_head(head_type, 2 * hidden_size, num_labels, dropout)

        # Loss config
        self.loss_type = str(loss_cfg["type"]).lower().strip()

        cw = None
        class_weights = loss_cfg.get("class_weights")
        if class_weights is None:
            cw = None
        elif isinstance(class_weights, torch.Tensor):
            cw = class_weights.detach().float()
        else:
            s = str(class_weights).strip()

            if not s:
                cw = None
            elif s.startswith("[") and s.endswith("]"):
                vals = ast.literal_eval(s)          # parses "[1.0, 1.6, 1.2]"
                cw = torch.tensor([float(v) for v in vals], dtype=torch.float)
            else:
                cw = torch.tensor(
                    [float(x.strip()) for x in s.split(",") if x.strip()],
                    dtype=torch.float,
                )

        self.register_buffer("class_weights", cw)
        
        if self.loss_type in ["weighed_ce", "focal"]:
            print("Class weights:", cw)

        self.focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))

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
        labels = None,
        fusion_method: str = "concat",
    ):

        out_sent = self.encoder(input_ids=input_ids_sent, attention_mask=attention_mask_sent)
        out_term = self.encoder(input_ids=input_ids_term, attention_mask=attention_mask_term)

        cls_sent = out_sent.last_hidden_state[:, 0, :]
        cls_term = out_term.last_hidden_state[:, 0, :]

        fusion_method = fusion_method.lower().strip()

        if fusion_method == "sent":
            logits = self.head_single(self.dropout(cls_sent))

        elif fusion_method == "term":
            logits = self.head_single(self.dropout(cls_term))

        elif fusion_method == "concat":
            logits = self.head_concat(self.dropout(torch.cat([cls_sent, cls_term], dim=-1)))

        elif fusion_method == "add":
            logits = self.head_single(self.dropout(cls_sent + cls_term))

        elif fusion_method == "mul":
            logits = self.head_single(self.dropout(cls_sent * cls_term))

        elif fusion_method == "cross":
            q = out_term.last_hidden_state[:, 0:1, :]
            kpm = attention_mask_sent.eq(0)
            attn_out, _ = self.cross_attn(q, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm)
            logits = self.head_single(self.dropout(attn_out.squeeze(1)))

        elif fusion_method == "gated_concat":
            fused = gated_fusion_two(self.gate, cls_sent, cls_term)
            logits = self.head_single(self.dropout(fused))

        elif fusion_method == "bilinear":
            fused = bilinear_fusion(
                self.bilinear_proj_sent, self.bilinear_proj_term, self.bilinear_out, cls_sent, cls_term
            )
            logits = self.head_single(self.dropout(fused))

        elif fusion_method == "coattn":
            q_term = out_term.last_hidden_state[:, 0:1, :]
            q_sent = out_sent.last_hidden_state[:, 0:1, :]
            kpm_sent = attention_mask_sent.eq(0)
            kpm_term = attention_mask_term.eq(0)

            term_ctx, _ = self.coattn_term_to_sent(q_term, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm_sent)
            sent_ctx, _ = self.coattn_sent_to_term(q_sent, out_term.last_hidden_state, out_term.last_hidden_state, key_padding_mask=kpm_term)

            logits = self.head_single(self.dropout(term_ctx.squeeze(1) + sent_ctx.squeeze(1)))

        elif fusion_method == "late_interaction":
            sent_tok = out_sent.last_hidden_state  # [B, Ls, H]
            term_tok = out_term.last_hidden_state  # [B, Lt, H]

            sent_tok = torch.nn.functional.normalize(sent_tok, p=2, dim=-1)
            term_tok = torch.nn.functional.normalize(term_tok, p=2, dim=-1)

            sim = torch.matmul(term_tok, sent_tok.transpose(1, 2))  # [B, Lt, Ls]

            if attention_mask_sent is not None:
                mask = attention_mask_sent.unsqueeze(1).eq(0)  # [B, 1, Ls]
                sim = sim.masked_fill(mask.bool(), torch.finfo(sim.dtype).min)

            max_sim = sim.max(dim=-1).values  # [B, Lt]

            if attention_mask_term is not None:
                term_valid = attention_mask_term.float()
                pooled = masked_mean(max_sim, term_valid, dim=1)
            else:
                pooled = max_sim.mean(dim=1)

            cond = self.gate(torch.cat([cls_sent, cls_term], dim=-1))  # [B, H]
            fused = cond * pooled.unsqueeze(-1)
            logits = self.head_single(self.dropout(fused))

        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

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
