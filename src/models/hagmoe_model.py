from __future__ import annotations

import ast
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.core.loss.focal_loss import FocalLoss
from src.models.components.heads import build_head
from src.models.components.experts import FFNExpert
from src.models.components.gating import gated_fusion_three, topk_renorm
from src.models.components.fusion import bilinear_fusion_three
from src.models.components.pooling import masked_mean


class HAGMoE(nn.Module):
    def __init__(
        self,
        *,
        encoder=None,
        model_cfg: dict,
        loss_cfg: dict,
        moe_cfg: dict,
    ) -> None:
        super().__init__()

        if encoder is None:
            encoder_name = model_cfg["encoder"]["pretrained_name"]
            self.encoder = AutoModel.from_pretrained(encoder_name)
        elif isinstance(encoder, str):
            self.encoder = AutoModel.from_pretrained(encoder)
        else:
            self.encoder = encoder

        hidden_size = int(self.encoder.config.hidden_size)
        num_labels = int(model_cfg["common"]["num_labels"])
        fusion_cfg = model_cfg.get("fusion", {})
        head_cfg = model_cfg.get("head", {})
        hag_cfg = model_cfg.get("hagmoe", {})
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

        self.opinion_q = nn.Linear(hidden_size, hidden_size, bias=False)

        self.fusion_concat = nn.Sequential(
            nn.LayerNorm(3 * hidden_size),
            nn.Dropout(dropout),
            nn.Linear(3 * hidden_size, hidden_size),
        )

        self.gate = nn.Linear(3 * hidden_size, hidden_size)

        bilinear_rank = max(32, min(256, hidden_size // 4))
        self.bilinear_proj_sent = nn.Linear(hidden_size, bilinear_rank)
        self.bilinear_proj_term = nn.Linear(hidden_size, bilinear_rank)
        self.bilinear_out = nn.Linear(bilinear_rank, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.classifier = build_head(head_type, hidden_size, num_labels, dropout)

        self.num_groups = int(hag_cfg.get("num_groups", 3))
        self.num_experts = int(hag_cfg.get("experts_per_group", moe_cfg.get("num_experts", 8)))
        self.router_temperature = float(
            hag_cfg.get("router_temperature", moe_cfg.get("router", {}).get("temperature", 1.0))
        )
        self.group_temperature = float(hag_cfg.get("group_temperature", 1.0))
        if self.group_temperature <= 0:
            self.group_temperature = 1.0
        self.group_temperature_anneal = str(hag_cfg.get("group_temperature_anneal", "")).strip()

        self.hag_merge = str(hag_cfg.get("merge", "residual")).lower().strip()
        self.hag_fusion_method = str(hag_cfg.get("fusion_method", "")).strip()
        self.hag_use_group_loss = bool(hag_cfg.get("use_group_loss", False))
        self.hag_use_balance_loss = bool(hag_cfg.get("use_balance_loss", False))
        self.hag_use_diversity_loss = bool(hag_cfg.get("use_diversity_loss", False))
        self.hag_lambda_group = float(hag_cfg.get("lambda_group", 0.5))
        self.hag_lambda_balance = float(hag_cfg.get("lambda_balance", 0.01))
        self.hag_lambda_diversity = float(hag_cfg.get("lambda_diversity", 0.1))
        self.router_entropy_weight = float(moe_cfg.get("router", {}).get("entropy_weight", 0.0))
        self.router_entropy_target = moe_cfg.get("router", {}).get("entropy_target")
        self.router_collapse_weight = float(moe_cfg.get("router", {}).get("collapse_weight", 0.0))
        self.router_collapse_tau = float(moe_cfg.get("router", {}).get("collapse_tau", 0.02))
        router_topk_groups = hag_cfg.get("router_topk_groups")
        if router_topk_groups is None:
            router_topk_groups = hag_cfg.get("top_k")
        if router_topk_groups is None:
            router_topk_groups = moe_cfg.get("router", {}).get("top_k")
        self.router_topk_groups = int(router_topk_groups) if router_topk_groups is not None else 0
        self.router_topk_apply_in_eval = bool(hag_cfg.get("router_topk_apply_in_eval", True))
        self.hag_verbose_loss = bool(hag_cfg.get("verbose_loss", False))

        cfg = getattr(self.encoder, "config", None)
        intermediate_size = int(getattr(cfg, "intermediate_size", hidden_size * 4))
        hidden_act = str(getattr(cfg, "hidden_act", "gelu")).lower()
        act_fn = nn.GELU() if hidden_act == "gelu" else nn.ReLU()
        dropout_p = float(getattr(cfg, "hidden_dropout_prob", dropout))

        self.group_router = nn.Linear(hidden_size, self.num_groups)
        self.cond_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.expert_routers = nn.ModuleList(
            [nn.Linear(hidden_size, self.num_experts) for _ in range(self.num_groups)]
        )
        self.experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        FFNExpert(hidden_size, intermediate_size, dropout_p, act_fn)
                        for _ in range(self.num_experts)
                    ]
                )
                for _ in range(self.num_groups)
            ]
        )

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

        self.focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))

        labelid_to_groupid = None
        labels_cfg = model_cfg.get("labels", {})
        id2label = labels_cfg.get("id2label")
        label2id = labels_cfg.get("label2id")
        if self.hag_use_group_loss:
            if id2label is None and label2id is not None:
                id2label = {int(v): k for k, v in label2id.items()}
            if id2label is None:
                raise ValueError(
                    "hag_use_group_loss=True requires id2label/label2id for group mapping."
                )
            labelid_to_groupid = self._build_label_group_mapping(
                id2label=id2label,
                num_labels=num_labels,
                num_groups=self.num_groups,
            )
        self.register_buffer("labelid_to_groupid", labelid_to_groupid)

    @staticmethod
    def _neg_inf(dtype: torch.dtype) -> float:
        if dtype in (torch.float16, torch.bfloat16):
            return -1e4
        return -1e9

    @staticmethod
    def _parse_temperature_anneal(spec: str) -> tuple[float, float] | None:
        if not spec:
            return None
        sep = "," if "," in spec else (":" if ":" in spec else None)
        if sep is None:
            return None
        parts = [p.strip() for p in spec.split(sep) if p.strip()]
        if len(parts) < 2:
            return None
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            return None
        return start, end

    def maybe_update_group_temperature(
        self, *, epoch_idx: int | None, total_epochs: int | None
    ) -> None:
        """Anneal group temperature if a schedule is configured."""
        spec = self.group_temperature_anneal
        if not spec or epoch_idx is None or total_epochs is None:
            return
        parsed = self._parse_temperature_anneal(spec)
        if parsed is None:
            return
        start, end = parsed
        if total_epochs <= 1:
            ratio = 1.0
        else:
            ratio = float(epoch_idx) / float(max(1, total_epochs - 1))
            ratio = max(0.0, min(1.0, ratio))
        temp = start + (end - start) * ratio
        if temp <= 0:
            temp = 1e-6
        self.group_temperature = float(temp)

    @staticmethod
    def _build_label_group_mapping(
        *,
        id2label: dict,
        num_labels: int,
        num_groups: int,
    ) -> torch.Tensor:
        if num_groups < 3:
            raise ValueError("HAGMoE requires num_groups >= 3 for polarity-aware group loss")

        mapping = torch.full((int(num_labels),), -1, dtype=torch.long)
        for idx in range(int(num_labels)):
            name = str(id2label.get(idx, "")).lower().strip()
            # Group index order is fixed: 0=positive, 1=negative, 2=neutral.
            if name in {"positive", "pos", "posi"}:
                group_id = 0
            elif name in {"negative", "neg"}:
                group_id = 1
            elif name in {"neutral", "neu", "neut"}:
                group_id = 2
            else:
                group_id = -1
            if group_id >= 0:
                mapping[idx] = group_id

        if torch.any(mapping < 0):
            missing = [i for i in range(int(num_labels)) if int(mapping[i].item()) < 0]
            raise ValueError(
                "HAGMoE group loss requires id2label with positive/negative/neutral labels. "
                f"Unmapped label ids: {missing}"
            )
        return mapping

    @staticmethod
    def _gather_aspect_tokens(
        h_sent_tokens: torch.Tensor,
        aspect_mask_sent: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if aspect_mask_sent is None:
            return h_sent_tokens, None

        mask = aspect_mask_sent.to(dtype=torch.bool)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        lengths = mask.sum(dim=1)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        max_len = max(1, max_len)

        bsz, _, hidden = h_sent_tokens.shape
        term_tok = h_sent_tokens.new_zeros((bsz, max_len, hidden))
        term_attn_mask = mask.new_zeros((bsz, max_len), dtype=mask.dtype)

        for i in range(bsz):
            idx = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            take = h_sent_tokens[i, idx, :]
            take_len = min(max_len, int(take.size(0)))
            term_tok[i, :take_len, :] = take[:take_len]
            term_attn_mask[i, :take_len] = 1

        return term_tok, term_attn_mask

    def compute_opinion(
        self,
        h_sent_tokens: torch.Tensor,
        attention_mask_sent: torch.Tensor | None,
        h_aspect: torch.Tensor,
        aspect_mask_sent: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute opinion representation via attention over sentence tokens."""
        q = self.opinion_q(h_sent_tokens)
        scores = (q * h_aspect.unsqueeze(1)).sum(dim=-1) / math.sqrt(q.size(-1))

        if attention_mask_sent is not None:
            pad_mask = attention_mask_sent.eq(0)
            scores = scores.masked_fill(pad_mask, self._neg_inf(scores.dtype))

        if aspect_mask_sent is not None:
            aspect_mask = aspect_mask_sent.to(dtype=torch.bool)
            scores = scores.masked_fill(aspect_mask, self._neg_inf(scores.dtype))

        attn = torch.softmax(scores, dim=-1)
        h_opinion = torch.bmm(attn.unsqueeze(1), h_sent_tokens).squeeze(1)
        return h_opinion

    def _fusion_concat(
        self,
        *,
        rep_sent: torch.Tensor,
        h_aspect: torch.Tensor,
        h_opinion: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        fusion_in = torch.cat([rep_sent, h_aspect, h_opinion], dim=-1)
        if self.hag_verbose_loss and not hasattr(self, "_fusion_debug_printed"):
            self._fusion_debug_printed = True
            print(f"[HAGMoE] fusion=concat input={tuple(fusion_in.shape)}")
        return self.fusion_concat(fusion_in)

    @staticmethod
    def _fusion_add(
        *,
        rep_sent: torch.Tensor,
        h_aspect: torch.Tensor,
        h_opinion: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        return rep_sent + h_aspect + h_opinion

    @staticmethod
    def _fusion_mul(
        *,
        rep_sent: torch.Tensor,
        h_aspect: torch.Tensor,
        h_opinion: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        return rep_sent * h_aspect * h_opinion

    def _fusion_cross(
        self,
        *,
        h_aspect: torch.Tensor,
        h_opinion: torch.Tensor,
        out_sent,
        attention_mask_sent: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        q = h_aspect.unsqueeze(1)
        kpm = attention_mask_sent.eq(0)
        attn_out, _ = self.cross_attn(
            q, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm
        )
        q_o = h_opinion.unsqueeze(1)
        attn_out_o, _ = self.cross_attn(
            q_o, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm
        )
        return attn_out.squeeze(1) + attn_out_o.squeeze(1)

    def _fusion_gated_concat(
        self,
        *,
        rep_sent: torch.Tensor,
        h_aspect: torch.Tensor,
        h_opinion: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        return gated_fusion_three(
            self.gate,
            rep_sent,
            h_aspect,
            h_opinion,
            b_scale=0.5,
            c_scale=0.5,
        )

    def _fusion_bilinear(
        self,
        *,
        rep_sent: torch.Tensor,
        h_aspect: torch.Tensor,
        h_opinion: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        return bilinear_fusion_three(
            self.bilinear_proj_sent,
            self.bilinear_proj_term,
            self.bilinear_out,
            rep_sent,
            h_aspect,
            h_opinion,
        )

    def _fusion_coattn(
        self,
        *,
        out_sent,
        attention_mask_sent: torch.Tensor,
        aspect_mask_sent: torch.Tensor | None,
        **_: object,
    ) -> torch.Tensor:
        sent_tok = out_sent.last_hidden_state
        term_tok, term_attn_mask = self._gather_aspect_tokens(sent_tok, aspect_mask_sent)

        q_sent = sent_tok[:, 0:1, :]
        kpm_sent = attention_mask_sent.eq(0)
        kpm_term = term_attn_mask.eq(0) if term_attn_mask is not None else None

        term_ctx, _ = self.coattn_term_to_sent(
            term_tok, sent_tok, sent_tok, key_padding_mask=kpm_sent
        )
        if term_attn_mask is not None:
            term_ctx = masked_mean(term_ctx, term_attn_mask, dim=1)
        else:
            term_ctx = term_ctx.mean(dim=1)

        sent_ctx, _ = self.coattn_sent_to_term(
            q_sent, term_tok, term_tok, key_padding_mask=kpm_term
        )
        return term_ctx + sent_ctx.squeeze(1)

    def _fusion_late_interaction(
        self,
        *,
        rep_sent: torch.Tensor,
        h_aspect: torch.Tensor,
        h_opinion: torch.Tensor,
        out_sent,
        attention_mask_sent: torch.Tensor,
        aspect_mask_sent: torch.Tensor | None,
        **_: object,
    ) -> torch.Tensor:
        sent_tok = out_sent.last_hidden_state
        term_tok, term_attn_mask = self._gather_aspect_tokens(sent_tok, aspect_mask_sent)

        sent_tok = torch.nn.functional.normalize(sent_tok, p=2, dim=-1)
        term_tok = torch.nn.functional.normalize(term_tok, p=2, dim=-1)

        sim = torch.matmul(term_tok, sent_tok.transpose(1, 2))
        if attention_mask_sent is not None:
            mask = attention_mask_sent.unsqueeze(1).eq(0)
            sim = sim.masked_fill(mask.bool(), self._neg_inf(sim.dtype))
        if term_attn_mask is not None:
            term_mask = term_attn_mask.unsqueeze(-1).eq(0)
            sim = sim.masked_fill(term_mask.bool(), self._neg_inf(sim.dtype))

        max_sim = sim.max(dim=-1).values
        if term_attn_mask is not None:
            term_valid = term_attn_mask.float()
            weights = torch.softmax(max_sim, dim=-1)
            weights = weights * term_valid
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            weights = weights / denom
        else:
            weights = torch.softmax(max_sim, dim=-1)

        h_late = (weights.unsqueeze(-1) * term_tok).sum(dim=1)
        gate = torch.sigmoid(self.gate(torch.cat([rep_sent, h_aspect, h_opinion], dim=-1)))
        return gate * rep_sent + (1.0 - gate) * h_late

    def build_fusion(
        self,
        *,
        h_sent: torch.Tensor,
        h_aspect: torch.Tensor,
        h_opinion: torch.Tensor,
        out_sent,
        attention_mask_sent: torch.Tensor,
        aspect_mask_sent: torch.Tensor | None,
        fusion_method: str,
    ) -> torch.Tensor:
        """Dispatch fusion based on fusion_method."""
        fusion_method = fusion_method.lower().strip()
        rep_sent = h_sent

        if fusion_method in {"sent", "term"}:
            raise ValueError(
                "HAGMoE does not support fusion_method 'sent' or 'term'. "
                "Use one of: concat, add, mul, cross, gated_concat, bilinear, coattn, late_interaction."
            )
        fusion_map = {
            "concat": self._fusion_concat,
            "add": self._fusion_add,
            "mul": self._fusion_mul,
            "cross": self._fusion_cross,
            "gated_concat": self._fusion_gated_concat,
            "bilinear": self._fusion_bilinear,
            "coattn": self._fusion_coattn,
            "late_interaction": self._fusion_late_interaction,
        }
        if fusion_method not in fusion_map:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

        return fusion_map[fusion_method](
            rep_sent=rep_sent,
            h_aspect=h_aspect,
            h_opinion=h_opinion,
            out_sent=out_sent,
            attention_mask_sent=attention_mask_sent,
            aspect_mask_sent=aspect_mask_sent,
        )

    def route_group(self, h_fused: torch.Tensor) -> torch.Tensor:
        return self.group_router(h_fused)

    def _pool_aspect(
        self,
        hidden_states: torch.Tensor,
        aspect_mask: torch.Tensor | None,
        h_sent: torch.Tensor,
    ) -> torch.Tensor:
        if aspect_mask is None:
            return h_sent

        mask = aspect_mask.to(dtype=hidden_states.dtype)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.size(1) != hidden_states.size(1):
            mask = mask[:, : hidden_states.size(1)]

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (hidden_states * mask.unsqueeze(-1)).sum(dim=1) / denom

        has_span = mask.sum(dim=1) > 0
        if torch.any(~has_span):
            pooled = torch.where(has_span.unsqueeze(-1), pooled, h_sent)

        return pooled

    def _build_aspect_mask(
        self,
        *,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        aspect_start: torch.Tensor | None,
        aspect_end: torch.Tensor | None,
        aspect_mask_sent: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Build aspect mask with fallbacks: provided -> span -> term match."""
        mask = self._mask_from_provided(aspect_mask_sent)
        if mask is not None:
            return mask

        mask = self._mask_from_span(input_ids_sent, aspect_start, aspect_end)
        if mask is not None:
            return mask

        if input_ids_term is None:
            return None

        return self._mask_from_term_match(
            input_ids_sent=input_ids_sent,
            attention_mask_sent=attention_mask_sent,
            input_ids_term=input_ids_term,
        )

    @staticmethod
    def _mask_from_provided(aspect_mask_sent: torch.Tensor | None) -> torch.Tensor | None:
        if aspect_mask_sent is None:
            return None
        return aspect_mask_sent

    @staticmethod
    def _mask_from_span(
        input_ids_sent: torch.Tensor,
        aspect_start: torch.Tensor | None,
        aspect_end: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if aspect_start is None or aspect_end is None:
            return None

        start = aspect_start.to(device=input_ids_sent.device)
        end = aspect_end.to(device=input_ids_sent.device)
        if start.dim() == 0:
            start = start.unsqueeze(0)
        if end.dim() == 0:
            end = end.unsqueeze(0)
        L = input_ids_sent.size(1)
        positions = torch.arange(L, device=input_ids_sent.device).unsqueeze(0)
        mask = (positions >= start.unsqueeze(1)) & (positions < end.unsqueeze(1))
        return mask.to(dtype=torch.long)

    def _mask_from_term_match(
        self,
        *,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
    ) -> torch.Tensor | None:
        """Match term tokens inside sentence content span to build aspect mask."""

        cls_id = getattr(self.encoder.config, "cls_token_id", None)
        sep_id = getattr(self.encoder.config, "sep_token_id", None)
        pad_id = getattr(self.encoder.config, "pad_token_id", None)
        special_ids = {x for x in (cls_id, sep_id, pad_id) if x is not None}

        bsz, L = input_ids_sent.shape
        mask_out = torch.zeros((bsz, L), device=input_ids_sent.device, dtype=torch.long)

        for i in range(bsz):
            sent_ids = input_ids_sent[i].tolist()
            sent_mask = attention_mask_sent[i].tolist()
            valid_len = int(sum(sent_mask))
            if valid_len <= 0:
                continue
            # Strip [CLS] token; stop at valid_len (exclude padding).
            content_start = 1
            content_end = valid_len
            if (
                content_end > content_start
                and sep_id is not None
                and sent_ids[content_end - 1] == sep_id
            ):
                # Strip trailing [SEP] at the end of valid content.
                content_end -= 1
            if content_end < content_start:
                content_end = content_start
            content_ids = sent_ids[content_start:content_end]

            term_ids_full = input_ids_term[i].tolist()
            term_ids = [tid for tid in term_ids_full if tid not in special_ids]
            if not term_ids:
                continue

            match_idx = -1
            for j in range(len(content_ids) - len(term_ids) + 1):
                if content_ids[j : j + len(term_ids)] == term_ids:
                    match_idx = j
                    break
            if match_idx < 0:
                continue

            start = content_start + match_idx
            end = start + len(term_ids)
            if start >= L:
                continue
            end = min(end, L)
            mask_out[i, start:end] = 1

        return mask_out

    def apply_grouped_experts(
        self, h_fused: torch.Tensor, h_expert_in: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
    ]:
        """Apply grouped experts with soft routing over groups."""
        group_logits = self.route_group(h_fused) / self.group_temperature
        p_group_raw = torch.softmax(group_logits, dim=-1)
        apply_topk = self.router_topk_groups > 0 and (self.training or self.router_topk_apply_in_eval)
        if apply_topk:
            # Top-k soft routing at group level: mask + renormalize to preserve probability simplex.
            # Example (G=3, k=2): [0.6, 0.3, 0.1] -> [0.6, 0.3, 0.0] -> [0.6667, 0.3333, 0.0]
            p_group = topk_renorm(p_group_raw, self.router_topk_groups)
            if self.hag_verbose_loss and not hasattr(self, "_topk_checked"):
                self._topk_checked = True
                with torch.no_grad():
                    sums = p_group.sum(dim=-1)
                    ones = torch.ones_like(sums)
                    if not torch.allclose(sums, ones, atol=1e-4, rtol=1e-4):
                        raise AssertionError("HAGMoE top-k routing: group probs do not sum to 1.")
                    nonzero = (p_group > 0).sum(dim=-1)
                    if torch.any(nonzero > self.router_topk_groups):
                        raise AssertionError("HAGMoE top-k routing: more than k groups are non-zero.")
                if self.hag_verbose_loss and not hasattr(self, "_topk_logged"):
                    self._topk_logged = True
                    with torch.no_grad():
                        gm_raw = p_group_raw.mean(dim=0)
                        gm_used = p_group.mean(dim=0)
                        nz = (p_group > 0).sum(dim=-1).float()
                        print(
                            "[HAGMoE][TopK] "
                            f"k={self.router_topk_groups} "
                            f"group_mean_raw="
                            f"{' '.join([f'g{i}={gm_raw[i].item():.6f}' for i in range(gm_raw.numel())])} "
                            f"group_mean_topk="
                            f"{' '.join([f'g{i}={gm_used[i].item():.6f}' for i in range(gm_used.numel())])} "
                            f"nonzero_groups_mean={nz.mean().item():.2f}"
                        )
        else:
            p_group = p_group_raw
        if self.hag_verbose_loss and self.training and not hasattr(self, "_group_temp_logged"):
            print(f"[HAGMoE] group_temperature={self.group_temperature:.4f}")
            self._group_temp_logged = True

        h_moe = torch.zeros_like(h_fused)
        p_expert_list: list[torch.Tensor] = []
        expert_outs_list: list[torch.Tensor] = []
        for g in range(self.num_groups):
            logits = self.expert_routers[g](h_expert_in) / self.router_temperature
            p_expert = torch.softmax(logits, dim=-1)
            p_expert_list.append(p_expert)

            expert_outs = [expert(h_expert_in) for expert in self.experts[g]]
            expert_stack = torch.stack(expert_outs, dim=1)
            expert_outs_list.append(expert_stack)

            h_g = (p_expert.unsqueeze(-1) * expert_stack).sum(dim=1)
            h_moe = h_moe + p_group[:, g].unsqueeze(-1) * h_g

        return h_moe, group_logits, p_group, p_expert_list, expert_outs_list, p_group_raw

    def _compute_fused_and_cond(
        self,
        *,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        aspect_start: torch.Tensor | None,
        aspect_end: torch.Tensor | None,
        aspect_mask_sent: torch.Tensor | None,
        fusion_method: str,
    ):
        """Compute fused representation and conditioned expert input."""
        out_sent = self.encoder(input_ids=input_ids_sent, attention_mask=attention_mask_sent)

        h_sent = out_sent.last_hidden_state[:, 0, :]
        aspect_mask = self._build_aspect_mask(
            input_ids_sent=input_ids_sent,
            attention_mask_sent=attention_mask_sent,
            input_ids_term=input_ids_term,
            aspect_start=aspect_start,
            aspect_end=aspect_end,
            aspect_mask_sent=aspect_mask_sent,
        )
        h_aspect = self._pool_aspect(out_sent.last_hidden_state, aspect_mask, h_sent)

        h_opinion = self.compute_opinion(
            out_sent.last_hidden_state, attention_mask_sent, h_aspect, aspect_mask
        )

        fusion_arg = str(fusion_method).strip() if fusion_method is not None else ""
        effective_fusion = fusion_arg if fusion_arg else str(self.hag_fusion_method or "").strip()
        if not effective_fusion:
            effective_fusion = "concat"
        self._last_fusion_method = effective_fusion

        h_fused = self.build_fusion(
            h_sent=h_sent,
            h_aspect=h_aspect,
            h_opinion=h_opinion,
            out_sent=out_sent,
            attention_mask_sent=attention_mask_sent,
            aspect_mask_sent=aspect_mask,
            fusion_method=effective_fusion,
        )

        h_expert_in = self.cond_proj(torch.cat([h_fused, h_aspect], dim=-1))

        return out_sent, h_fused, h_aspect, h_expert_in, aspect_mask

    def forward(
        self,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        aspect_start: torch.Tensor | None = None,
        aspect_end: torch.Tensor | None = None,
        aspect_mask_sent: torch.Tensor | None = None,
        labels=None,
        fusion_method: str = "concat",
    ) -> Dict[str, Any]:
        """Forward pass returning logits and loss breakdowns."""
        out_sent, h_fused, h_aspect, h_expert_in, aspect_mask = self._compute_fused_and_cond(
            input_ids_sent=input_ids_sent,
            attention_mask_sent=attention_mask_sent,
            input_ids_term=input_ids_term,
            attention_mask_term=attention_mask_term,
            aspect_start=aspect_start,
            aspect_end=aspect_end,
            aspect_mask_sent=aspect_mask_sent,
            fusion_method=fusion_method,
        )

        (
            h_moe,
            group_logits,
            p_group,
            p_expert_list,
            expert_outs_list,
            p_group_raw,
        ) = self.apply_grouped_experts(h_fused, h_expert_in)
        self._last_group_probs = p_group.detach()
        self._last_group_probs_raw = p_group_raw.detach()
        self._last_expert_probs = [p.detach() for p in p_expert_list]
        self._last_h_expert_in = h_expert_in.detach()
        if self.hag_merge == "moe_only":
            h_final = h_moe
        else:
            h_final = h_fused + h_moe
        logits = self.classifier(self.dropout(h_final))

        out = self._compute_loss(
            logits,
            labels,
            group_logits=group_logits,
            p_group=p_group,
            p_expert_list=p_expert_list,
            expert_outs_list=expert_outs_list,
        )
        out["group_probs"] = p_group.detach()
        out["group_probs_raw"] = p_group_raw.detach()
        out["moe_stats"] = {
            "group_probs": p_group.detach(),
            "group_probs_raw": p_group_raw.detach(),
            "expert_probs": [p.detach() for p in p_expert_list],
        }

        if self.training and self.hag_verbose_loss and labels is not None and self._should_print_debug():
            if not hasattr(self, "_expert_input_logged"):
                print("[HAGMoE] expert_input=conditioned (h_cond)")
                self._expert_input_logged = True
            print(
                "[HAGMoE] "
                f"main={out['loss_main'].item():.4f} "
                f"group_raw={(out['loss_group_raw'].item() if out['loss_group_raw'] is not None else 0.0):.4f} "
                f"group_used={(out['loss_group_used'].item() if out['loss_group_used'] is not None else 0.0):.4f} "
                f"balance_raw={(out['loss_balance_raw'].item() if out['loss_balance_raw'] is not None else 0.0):.4f} "
                f"balance_used={(out['loss_balance_used'].item() if out['loss_balance_used'] is not None else 0.0):.4f} "
                f"div_raw={(out['loss_diversity_raw'].item() if out['loss_diversity_raw'] is not None else 0.0):.4f} "
                f"div_used={(out['loss_diversity_used'].item() if out['loss_diversity_used'] is not None else 0.0):.4f} "
                f"ent_raw={(out['loss_entropy_raw'].item() if out.get('loss_entropy_raw') is not None else 0.0):.4f} "
                f"ent_used={(out['loss_entropy_used'].item() if out.get('loss_entropy_used') is not None else 0.0):.4f} "
                f"col_raw={(out['loss_collapse_raw'].item() if out.get('loss_collapse_raw') is not None else 0.0):.4f} "
                f"col_used={(out['loss_collapse_used'].item() if out.get('loss_collapse_used') is not None else 0.0):.4f}"
            )

        return out

    def _should_print_debug(self) -> bool:
        step = int(getattr(self, "_dbg_step", 50) or 50)
        if step <= 0:
            return False
        count = int(getattr(self, "_dbg_counter", 0)) + 1
        self._dbg_counter = count
        return (count % step) == 0

    def _compute_loss(
        self,
        logits,
        labels,
        *,
        group_logits: torch.Tensor | None = None,
        p_group: torch.Tensor | None = None,
        p_expert_list: list[torch.Tensor] | None = None,
        expert_outs_list: list[torch.Tensor] | None = None,
    ) -> Dict[str, Any]:
        """Compute total loss and all auxiliary loss components."""
        if labels is None:
            return self._empty_loss_outputs(logits)

        loss_main = self._loss_main(logits, labels)
        loss_group, _ = self._loss_group(group_logits, labels)
        loss_balance = self._loss_balance(p_expert_list, p_group)
        loss_diversity, div_pair_mean, div_pair_max, div_usage_stats = self._loss_diversity(
            expert_outs_list, p_expert_list
        )
        (
            loss_entropy,
            mean_ent,
            ent_of_mean,
            group_mean,
            min_group_mean,
            loss_collapse,
        ) = self._loss_entropy_collapse(p_group)

        lambda_group = float(self.hag_lambda_group) if self.hag_use_group_loss else 0.0
        lambda_balance = float(self.hag_lambda_balance) if self.hag_use_balance_loss else 0.0
        lambda_diversity = float(self.hag_lambda_diversity) if self.hag_use_diversity_loss else 0.0
        lambda_entropy = float(self.router_entropy_weight)
        lambda_collapse = float(self.router_collapse_weight)

        use_group = loss_group is not None and lambda_group > 0.0
        use_balance = loss_balance is not None and lambda_balance > 0.0
        use_diversity = loss_diversity is not None and lambda_diversity > 0.0
        use_entropy = loss_entropy is not None and lambda_entropy > 0.0
        use_collapse = loss_collapse is not None and lambda_collapse > 0.0

        aux_loss = torch.zeros((), device=logits.device)
        if use_group:
            aux_loss = aux_loss + lambda_group * loss_group
        if use_balance:
            aux_loss = aux_loss + lambda_balance * loss_balance
        if use_diversity:
            aux_loss = aux_loss + lambda_diversity * loss_diversity
        if use_entropy:
            aux_loss = aux_loss + lambda_entropy * loss_entropy
        if use_collapse:
            aux_loss = aux_loss + lambda_collapse * loss_collapse

        loss = loss_main + aux_loss

        return self._pack_loss_outputs(
            logits=logits,
            loss=loss,
            loss_main=loss_main,
            aux_loss=aux_loss,
            loss_group=loss_group,
            loss_balance=loss_balance,
            loss_diversity=loss_diversity,
            loss_entropy=loss_entropy,
            loss_collapse=loss_collapse,
            lambda_group=lambda_group,
            lambda_balance=lambda_balance,
            lambda_diversity=lambda_diversity,
            lambda_entropy=lambda_entropy,
            lambda_collapse=lambda_collapse,
            use_group=use_group,
            use_balance=use_balance,
            use_diversity=use_diversity,
            use_entropy=use_entropy,
            use_collapse=use_collapse,
            mean_ent=mean_ent,
            ent_of_mean=ent_of_mean,
            group_mean=group_mean,
            min_group_mean=min_group_mean,
            div_pair_mean=div_pair_mean,
            div_pair_max=div_pair_max,
            div_usage_stats=div_usage_stats,
        )

    def _empty_loss_outputs(self, logits: torch.Tensor) -> Dict[str, Any]:
        return {
            "loss": None,
            "logits": logits,
            "loss_main": None,
            "aux_loss": None,
            "loss_group": None,
            "loss_balance": None,
            "loss_diversity": None,
            "loss_entropy_raw": None,
            "loss_entropy_used": None,
            "loss_collapse_raw": None,
            "loss_collapse_used": None,
            "loss_group_raw": None,
            "loss_balance_raw": None,
            "loss_diversity_raw": None,
            "loss_group_used": None,
            "loss_balance_used": None,
            "loss_diversity_used": None,
            "loss_lambda": None,
            "lambda_group": float(self.hag_lambda_group),
            "lambda_balance": float(self.hag_lambda_balance),
            "lambda_diversity": float(self.hag_lambda_diversity),
            "lambda_entropy": float(self.router_entropy_weight),
            "lambda_collapse": float(self.router_collapse_weight),
            "mean_ent": None,
            "ent_of_mean": None,
            "group_mean": None,
            "min_group_mean": None,
            "collapse_tau": float(self.router_collapse_tau),
            "enabled_flags": {
                "group": False,
                "balance": False,
                "diversity": False,
                "entropy": False,
                "collapse": False,
            },
        }

    def _loss_main(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "ce":
            return F.cross_entropy(logits, labels)
        if self.loss_type == "weighted_ce":
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            return F.cross_entropy(logits, labels, weight=w)
        if self.loss_type == "focal":
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss_fn = FocalLoss(gamma=self.focal_gamma, alpha=w, reduction="mean")
            return loss_fn(logits, labels)
        raise RuntimeError(f"Unexpected loss_type: {self.loss_type}")

    def _loss_group(
        self, group_logits: torch.Tensor | None, labels: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.hag_use_group_loss or group_logits is None:
            return None, None
        if self.labelid_to_groupid is None:
            raise RuntimeError(
                "hag_use_group_loss=True but labelid_to_groupid is not set. "
                "Provide id2label/label2id in cfg."
            )
        group_target = self.labelid_to_groupid[labels]
        loss_group = F.cross_entropy(group_logits, group_target)
        if self.hag_verbose_loss and self.training and not hasattr(self, "_group_targets_logged"):
            counts = torch.bincount(group_target, minlength=self.num_groups)
            print(
                "[HAGMoE] group_targets "
                f"min={int(group_target.min().item())} "
                f"max={int(group_target.max().item())} "
                f"counts={counts.tolist()}"
            )
            self._group_targets_logged = True
        return loss_group, group_target

    @staticmethod
    def _loss_balance(
        p_expert_list: list[torch.Tensor] | None,
        p_group: torch.Tensor | None,
    ) -> Optional[torch.Tensor]:
        if p_expert_list is None or len(p_expert_list) == 0:
            return None
        losses = []
        for g, p_expert in enumerate(p_expert_list):
            if p_expert.numel() == 0:
                continue
            p_mean = p_expert.mean(dim=0)
            assign = torch.argmax(p_expert.detach(), dim=-1)
            f = F.one_hot(assign, num_classes=p_expert.size(1)).float().mean(dim=0)
            n_experts = p_expert.size(1)
            balance_g = n_experts * torch.sum(f * p_mean)
            if p_group is not None and p_group.numel() > 0 and g < p_group.size(1):
                w_g = p_group[:, g].mean().detach()
                balance_g = w_g * balance_g
            losses.append(balance_g)
        if not losses:
            return None
        if p_group is not None and p_group.numel() > 0:
            return sum(losses)
        return sum(losses) / len(losses)

    @staticmethod
    def _loss_diversity(
        expert_outs_list: list[torch.Tensor] | None,
        p_expert_list: list[torch.Tensor] | None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], list]:
        loss_diversity = None
        div_pair_mean = None
        div_pair_max = None
        div_usage_stats = []
        if expert_outs_list is None or p_expert_list is None:
            return loss_diversity, div_pair_mean, div_pair_max, div_usage_stats

        losses = []
        pair_means = []
        pair_maxes = []
        for expert_stack, p_expert in zip(expert_outs_list, p_expert_list):
            if expert_stack is None or p_expert is None:
                continue
            if expert_stack.dim() != 3:
                continue
            bsz, n_experts, _ = expert_stack.shape
            if n_experts < 2 or bsz == 0:
                continue
            # Expert outputs: [B, E, d] -> normalize per sample.
            z = F.normalize(expert_stack, p=2, dim=-1)
            dot = torch.einsum("bed,bfd->bef", z, z).mean(dim=0)
            dot = dot.clamp(min=-0.999, max=0.999)
            offdiag = torch.triu(dot, diagonal=1)
            offdiag_vals = offdiag[offdiag != 0]
            if offdiag_vals.numel() == 0:
                continue

            w = p_expert.mean(dim=0).detach()
            div_usage_stats.append(
                {
                    "w_mean": float(w.mean().item()),
                    "w_min": float(w.min().item()),
                    "w_max": float(w.max().item()),
                }
            )
            w_outer = torch.outer(w, w)
            w_offdiag = torch.triu(w_outer, diagonal=1)
            weighted = (w_offdiag * (offdiag ** 2)).sum()
            num_pairs = max(n_experts * (n_experts - 1) // 2, 1)
            losses.append(weighted / float(num_pairs))
            pair_means.append(offdiag_vals.abs().mean())
            pair_maxes.append(offdiag_vals.abs().max())

        if losses:
            loss_diversity = sum(losses) / len(losses)
            div_pair_mean = sum(pair_means) / len(pair_means)
            div_pair_max = max(pair_maxes)
        return loss_diversity, div_pair_mean, div_pair_max, div_usage_stats

    def _loss_entropy_collapse(
        self, p_group: torch.Tensor | None
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if p_group is None or p_group.numel() == 0:
            return None, None, None, None, None, None
        p = p_group.clamp_min(1e-12)
        mean_ent = -(p * p.log()).sum(dim=-1).mean()
        if self.router_entropy_target is None:
            loss_entropy = -mean_ent
        else:
            target = torch.as_tensor(
                float(self.router_entropy_target), device=mean_ent.device, dtype=mean_ent.dtype
            )
            loss_entropy = (mean_ent - target) ** 2
        group_mean = p_group.mean(dim=0)
        min_group_mean = group_mean.min()
        mean_p = group_mean.clamp_min(1e-12)
        ent_of_mean = -(mean_p * mean_p.log()).sum()
        loss_collapse = F.relu(self.router_collapse_tau - min_group_mean).pow(2)
        return loss_entropy, mean_ent, ent_of_mean, group_mean, min_group_mean, loss_collapse

    def _pack_loss_outputs(
        self,
        *,
        logits: torch.Tensor,
        loss: torch.Tensor,
        loss_main: torch.Tensor,
        aux_loss: torch.Tensor,
        loss_group: torch.Tensor | None,
        loss_balance: torch.Tensor | None,
        loss_diversity: torch.Tensor | None,
        loss_entropy: torch.Tensor | None,
        loss_collapse: torch.Tensor | None,
        lambda_group: float,
        lambda_balance: float,
        lambda_diversity: float,
        lambda_entropy: float,
        lambda_collapse: float,
        use_group: bool,
        use_balance: bool,
        use_diversity: bool,
        use_entropy: bool,
        use_collapse: bool,
        mean_ent: torch.Tensor | None,
        ent_of_mean: torch.Tensor | None,
        group_mean: torch.Tensor | None,
        min_group_mean: torch.Tensor | None,
        div_pair_mean: torch.Tensor | None,
        div_pair_max: torch.Tensor | None,
        div_usage_stats: list,
    ) -> Dict[str, Any]:
        loss_entropy_raw = loss_entropy.detach() if loss_entropy is not None else None
        if loss_entropy is not None:
            loss_entropy_used = loss_entropy.detach() * lambda_entropy
        else:
            loss_entropy_used = None
        if loss_entropy_used is None and lambda_entropy == 0.0:
            loss_entropy_used = torch.zeros((), device=logits.device)

        loss_collapse_raw = loss_collapse.detach() if loss_collapse is not None else None
        if loss_collapse is not None:
            loss_collapse_used = loss_collapse.detach() * lambda_collapse
        else:
            loss_collapse_used = None
        if loss_collapse_used is None and lambda_collapse == 0.0:
            loss_collapse_used = torch.zeros((), device=logits.device)

        return {
            "loss": loss,
            "loss_total": loss.detach(),
            "logits": logits,
            "loss_main": loss_main.detach(),
            "aux_loss": aux_loss.detach(),
            "loss_group": loss_group.detach() if loss_group is not None else None,
            "loss_balance": loss_balance.detach() if loss_balance is not None else None,
            "loss_diversity": loss_diversity.detach() if loss_diversity is not None else None,
            "loss_entropy_raw": loss_entropy_raw,
            "loss_entropy_used": loss_entropy_used,
            "loss_collapse_raw": loss_collapse_raw,
            "loss_collapse_used": loss_collapse_used,
            "loss_group_raw": loss_group.detach() if loss_group is not None else None,
            "loss_balance_raw": loss_balance.detach() if loss_balance is not None else None,
            "loss_diversity_raw": loss_diversity.detach() if loss_diversity is not None else None,
            "loss_group_used": (loss_group.detach() * lambda_group) if use_group else None,
            "loss_balance_used": (loss_balance.detach() * lambda_balance) if use_balance else None,
            "loss_diversity_used": (loss_diversity.detach() * lambda_diversity)
            if use_diversity
            else None,
            "loss_lambda": aux_loss.detach(),
            "lambda_group": lambda_group,
            "lambda_balance": lambda_balance,
            "lambda_diversity": lambda_diversity,
            "lambda_entropy": lambda_entropy,
            "lambda_collapse": lambda_collapse,
            "mean_ent": mean_ent.detach() if mean_ent is not None else None,
            "ent_of_mean": ent_of_mean.detach() if ent_of_mean is not None else None,
            "group_mean": group_mean.detach() if group_mean is not None else None,
            "min_group_mean": min_group_mean.detach() if min_group_mean is not None else None,
            "collapse_tau": float(self.router_collapse_tau),
            "enabled_flags": {
                "group": bool(use_group),
                "balance": bool(use_balance),
                "diversity": bool(use_diversity),
                "entropy": bool(use_entropy),
                "collapse": bool(use_collapse),
            },
            "div_pair_mean": div_pair_mean.detach() if div_pair_mean is not None else None,
            "div_pair_max": div_pair_max.detach() if div_pair_max is not None else None,
            "div_usage_stats": div_usage_stats,
        }

    def _collect_aux_loss(self):
        return torch.zeros((), device=next(self.parameters()).device)

    @torch.no_grad()
    def print_moe_debug(self, topn: int = 3, eps_dead: float = 1e-6) -> None:
        """Print lightweight routing statistics for debugging."""
        if not hasattr(self, "_last_group_probs") or not hasattr(self, "_last_expert_probs"):
            print("[HAGMoE] No routing stats yet.")
            return

        group_probs = getattr(self, "_last_group_probs", None)
        expert_probs = getattr(self, "_last_expert_probs", None)
        if group_probs is None or expert_probs is None:
            print("[HAGMoE] No routing stats yet.")
            return

        group_mean = group_probs.mean(dim=0).detach().cpu()
        eps = 1e-12
        ent_of_mean = -torch.sum(group_mean * torch.log(group_mean.clamp_min(eps))).item()
        mean_ent = (
            -(group_probs * torch.log(group_probs.clamp_min(eps))).sum(dim=-1).mean().item()
        )
        group_pairs = " ".join([f"g{gi}={float(p):.6f}" for gi, p in enumerate(group_mean)])
        print("\n[HAGMoE Debug - Grouped Router]")
        if self.router_topk_groups > 0 and hasattr(self, "_last_group_probs_raw"):
            raw = getattr(self, "_last_group_probs_raw", None)
            if raw is not None and torch.is_tensor(raw) and raw.numel() > 0:
                gm_raw = raw.mean(dim=0).detach().cpu()
                gm_used = group_mean
                nz = (group_probs > 0).sum(dim=-1).float().mean().item()
                raw_pairs = " ".join([f"g{gi}={float(p):.6f}" for gi, p in enumerate(gm_raw)])
                used_pairs = " ".join([f"g{gi}={float(p):.6f}" for gi, p in enumerate(gm_used)])
                print(
                    f"  topk_groups={int(self.router_topk_groups)} "
                    f"group_mean_raw: {raw_pairs} "
                    f"group_mean_topk: {used_pairs} "
                    f"nonzero_groups_mean={nz:.2f}"
                )
        print(f"  group_mean: {group_pairs}")
        print(f"  ent_of_mean: {ent_of_mean:.6f}")
        print(f"  mean_ent: {mean_ent:.6f}")

        for g, p_expert in enumerate(expert_probs):
            if p_expert is None or p_expert.numel() == 0:
                print(f"  group{g}: no expert stats")
                continue

            usage = p_expert.mean(dim=0).detach().cpu()
            dead = int((usage < eps_dead).sum().item())
            topk = min(topn, usage.numel())
            botk = min(topn, usage.numel())
            topv, topi = torch.topk(usage, k=topk, largest=True)
            botv, boti = torch.topk(usage, k=botk, largest=False)
            top_pairs = " ".join([f"e{int(i)}={float(v):.6f}" for v, i in zip(topv, topi)])
            bot_pairs = " ".join([f"e{int(i)}={float(v):.6f}" for v, i in zip(botv, boti)])

            print(
                f"  group{g}: min={float(usage.min()):.6f} max={float(usage.max()):.6f} dead(<{eps_dead:g})={dead}"
            )
            print(f"    top: {top_pairs}")
            print(f"    bot: {bot_pairs}")
    
