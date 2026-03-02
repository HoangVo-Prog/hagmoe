from __future__ import annotations

import math
from typing import Optional, Dict, Any
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.loss.focal_loss import FocalLoss
from src.models.base_model import BaseModel


class _ExpertMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_labels: int, dropout_p: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class SeqMoELogits(nn.Module):
    """Sequence-level MoE that outputs delta logits.

    Input:  x [B, D]
    Output: delta_logits [B, C]

    Caches:
      last_router_logits: [B, E]
      last_topk_idx: [B, K]
    """

    def __init__(
        self,
        *,
        in_dim: int,
        num_labels: int,
        num_experts: int,
        top_k: int,
        dropout_p: float,
        expert_hidden: Optional[int],
        router_bias: bool,
        router_jitter: float,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.moe_top_k = int(top_k)

        hidden = int(expert_hidden or max(64, in_dim))
        self.router = nn.Linear(in_dim, self.num_experts, bias=bool(router_bias))
        self.dropout = nn.Dropout(float(dropout_p))
        self.experts = nn.ModuleList(
            [_ExpertMLP(in_dim, hidden, num_labels, float(dropout_p)) for _ in range(self.num_experts)]
        )

        self.last_router_logits: Optional[torch.Tensor] = None
        self.last_topk_idx: Optional[torch.Tensor] = None
        
        self.num_labels = num_labels
        self.router_jitter = router_jitter

    def set_top_k(self, k: int) -> None:
        k = int(k)
        if k < 1:
            raise ValueError("top_k must be >= 1")
        if k > self.num_experts:
            raise ValueError("top_k must be <= num_experts")
        self.moe_top_k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"SeqMoELogits expects [B, D], got {tuple(x.shape)}")

        # Router logits
        router_logits = self.router(x)  # [B, E]

        if self.training and self.router_jitter and self.router_jitter > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * float(self.router_jitter)

        # TopK
        k = int(self.moe_top_k)
        topk_vals, topk_idx = torch.topk(router_logits, k=k, dim=-1)  # [B, K]
        topk_w = torch.softmax(topk_vals, dim=-1)  # [B, K]

        # Cache for debug and aux loss
        self.last_router_logits = router_logits
        self.last_topk_idx = topk_idx

        # Weighted sum of expert outputs
        # Compute outputs only for selected experts (small E and K, simple loop)
        B = x.size(0)
        C = self.num_labels
        out = x.new_zeros((B, C))

        for j in range(k):
            idx_j = topk_idx[:, j]  # [B]
            w_j = topk_w[:, j].unsqueeze(-1)  # [B, 1]

            # Group samples by expert id to avoid per-sample expert calls
            for e in torch.unique(idx_j):
                e_int = int(e.item())
                mask = (idx_j == e)
                if not mask.any():
                    continue
                x_e = x[mask]
                y_e = self.experts[e_int](x_e)  # [n_e, C]
                if y_e.dtype != out.dtype:
                    y_e = y_e.to(out.dtype)
                out[mask] = out[mask] + y_e * w_j[mask].to(out.dtype)

        out = self.dropout(out)
        return out

    @torch.no_grad()
    def debug_stats(self) -> Optional[Dict[str, Any]]:
        if self.last_router_logits is None or self.last_topk_idx is None:
            return None
        logits = self.last_router_logits
        topk_idx = self.last_topk_idx
        E = int(self.num_experts)

        probs = torch.softmax(logits, dim=-1)
        usage_soft = probs.mean(dim=0)  # [E]

        flat = topk_idx.reshape(-1)
        counts = torch.bincount(flat, minlength=E).float()
        frac = counts / (counts.sum().clamp_min(1.0))

        return {
            "usage_soft": usage_soft.detach().cpu(),
            "topk_frac": frac.detach().cpu(),
            "moe_top_k": int(self.moe_top_k),
        }


class MoESkip(nn.Module):
    """
    MoE module with skip connection for sequence-level logits.
    Output: base_logits + beta * moe_delta_logits
    """
    def __init__(
        self,
        *,
        in_dim: int,
        num_labels: int,
        num_experts: int,
        top_k: int,
        dropout_p: float,
        expert_hidden: Optional[int],
        router_bias: bool,
        beta_start: float,
        beta_end: float,
        beta_warmup_steps: int,
        router_jitter: float
    ) -> None:
        super().__init__()
        
        self.in_dim = int(in_dim)
        self.num_labels = int(num_labels)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_warmup_steps = int(beta_warmup_steps)
        self._global_step = 0
        
        # Base classification head
        self.base_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout_p),
            nn.Linear(in_dim, num_labels),
        )
        
        self.moe_residual = SeqMoELogits(
            in_dim=in_dim,
            num_labels=num_labels,
            num_experts=int(num_experts),
            top_k=int(top_k),
            expert_hidden=expert_hidden,
            dropout_p=float(dropout_p),
            router_bias=bool(router_bias),
            router_jitter=router_jitter,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] representation
        Returns:
            logits: [B, C]
        """
        if x.dim() != 2:
            raise ValueError(f"Expected [B, D], got {tuple(x.shape)}")
        
        # Base logits
        logits_base = self.base_head(x)
        
        # MoE delta logits
        delta = self.moe_residual(x)
        
        # Skip connection with beta weighting
        beta = self._get_beta()
        logits = logits_base + beta * delta
        
        return logits
    
    def _get_beta(self) -> float:
        """Compute current beta based on warmup schedule."""
        if not self.training or self.beta_warmup_steps <= 0:
            return float(self.beta_end)
        
        t = min(1.0, float(self._global_step) / float(self.beta_warmup_steps))
        beta = self.beta_start + (self.beta_end - self.beta_start) * t
        return float(beta)
    
    def increment_step(self) -> None:
        """Increment global step for beta warmup."""
        if self.training:
            self._global_step += 1
    
    def set_top_k(self, k: int) -> None:
        """Set top-k for MoE routing."""
        self.moe_residual.set_top_k(k)


class EncoderWithMoESkip(nn.Module):
    """
    Wrapper that adds MoE skip connection modules on top of base encoder.
    Supports both H and 2H dimensional inputs.
    """
    def __init__(
        self,
        *,
        base_encoder: nn.Module,
        moe_sk_h: MoESkip,
        moe_sk_2h: MoESkip,
    ) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.moe_sk_h = moe_sk_h
        self.moe_sk_2h = moe_sk_2h
    
    def forward(self, *args, **kwargs):
        return self.base_encoder(*args, **kwargs)


class MoESkipModel(BaseModel):
    """
    Model with MoE skip connections for both H and 2H dimensional representations.
    Combines base model fusion methods with MoE residual learning.
    """
    def __init__(
        self,
        *,
        encoder=None,
        model_cfg: dict,
        loss_cfg: dict,
        moe_cfg: dict,
    ) -> None:
        super().__init__(encoder=encoder, model_cfg=model_cfg, loss_cfg=loss_cfg, moe_cfg=moe_cfg)
        
        router_cfg = moe_cfg.get("router", {})
        sk_cfg = model_cfg.get("sk", {})
        self.aux_loss_weight = float(moe_cfg["load_balance"].get("coef", 0.0))
        self.router_entropy_weight = float(router_cfg.get("entropy_weight", 0.0))
        self.aux_warmup_steps = int(router_cfg.get("aux_warmup_steps", 0))
        self.jitter_warmup_steps = int(router_cfg.get("jitter_warmup_steps", 0))
        self.jitter_end = float(router_cfg.get("jitter_end", 0.0))
        self._global_step = 0
        
        cfg = getattr(self.encoder, "config", None)
        hidden_size = int(getattr(cfg, "hidden_size"))
        self._jitter_start = float(router_cfg.get("noise_std", 0.0))
        
        expert_hidden = sk_cfg.get("expert_hidden")
        if expert_hidden is None:
            expert_hidden = max(64, hidden_size)
        
        # MoE skip connection for H-dimensional inputs
        moe_sk_h = MoESkip(
            in_dim=hidden_size,
            num_labels=int(model_cfg["common"]["num_labels"]),
            num_experts=int(moe_cfg["num_experts"]),
            top_k=int(moe_cfg["top_k"]),
            dropout_p=float(model_cfg["fusion"].get("dropout", model_cfg["encoder"].get("dropout", 0.1))),
            expert_hidden=expert_hidden,
            router_bias=bool(router_cfg.get("bias", True)),
            beta_start=float(sk_cfg.get("beta_start", 0.0)),
            beta_end=float(sk_cfg.get("beta_end", 1.0)),
            beta_warmup_steps=int(sk_cfg.get("beta_warmup_steps", 0)),
            router_jitter=float(router_cfg.get("noise_std", 0.0)),
        )
        
        # MoE skip connection for 2H-dimensional inputs (concat)
        moe_sk_2h = MoESkip(
            in_dim=2 * hidden_size,
            num_labels=int(model_cfg["common"]["num_labels"]),
            num_experts=int(moe_cfg["num_experts"]),
            top_k=int(moe_cfg["top_k"]),
            dropout_p=float(model_cfg["fusion"].get("dropout", model_cfg["encoder"].get("dropout", 0.1))),
            expert_hidden=expert_hidden * 2 if expert_hidden else None,
            router_bias=bool(router_cfg.get("bias", True)),
            beta_start=float(sk_cfg.get("beta_start", 0.0)),
            beta_end=float(sk_cfg.get("beta_end", 1.0)),
            beta_warmup_steps=int(sk_cfg.get("beta_warmup_steps", 0)),
            router_jitter=float(router_cfg.get("noise_std", 0.0)),
        )
        
        # Replace encoder with wrapped version
        base_encoder = self.encoder
        self.encoder = EncoderWithMoESkip(
            base_encoder=base_encoder,
            moe_sk_h=moe_sk_h,
            moe_sk_2h=moe_sk_2h,
        )
    
    def forward(
        self,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        labels=None,
        fusion_method: str = "concat",
    ):
        if self.training:
            self._global_step += 1
            # Update jitter for both MoE modules
            new_jitter = float(self._jitter())
            self.encoder.moe_sk_h.moe_residual.router_jitter = new_jitter
            self.encoder.moe_sk_2h.moe_residual.router_jitter = new_jitter
            # Increment steps for beta warmup
            self.encoder.moe_sk_h.increment_step()
            self.encoder.moe_sk_2h.increment_step()
        
        # Get base encoder outputs
        out_sent = self.encoder.base_encoder(input_ids=input_ids_sent, attention_mask=attention_mask_sent)
        out_term = self.encoder.base_encoder(input_ids=input_ids_term, attention_mask=attention_mask_term)
        
        cls_sent = out_sent.last_hidden_state[:, 0, :]
        cls_term = out_term.last_hidden_state[:, 0, :]
        
        fusion_method = fusion_method.lower().strip()
        
        # Compute representation based on fusion method
        if fusion_method == "sent":
            rep = cls_sent
            logits = self.encoder.moe_sk_h(self.dropout(rep))
        
        elif fusion_method == "term":
            rep = cls_term
            logits = self.encoder.moe_sk_h(self.dropout(rep))
        
        elif fusion_method == "concat":
            rep = torch.cat([cls_sent, cls_term], dim=-1)
            logits = self.encoder.moe_sk_2h(self.dropout(rep))
        
        elif fusion_method == "add":
            rep = cls_sent + cls_term
            logits = self.encoder.moe_sk_h(self.dropout(rep))
        
        elif fusion_method == "mul":
            rep = cls_sent * cls_term
            logits = self.encoder.moe_sk_h(self.dropout(rep))
        
        elif fusion_method == "cross":
            q = out_term.last_hidden_state[:, 0:1, :]
            kpm = attention_mask_sent.eq(0)
            attn_out, _ = self.cross_attn(q, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm)
            rep = attn_out.squeeze(1)
            logits = self.encoder.moe_sk_h(self.dropout(rep))
        
        elif fusion_method == "gated_concat":
            g = torch.sigmoid(self.gate(torch.cat([cls_sent, cls_term], dim=-1)))
            rep = g * cls_sent + (1 - g) * cls_term
            logits = self.encoder.moe_sk_h(self.dropout(rep))
        
        elif fusion_method == "bilinear":
            rep = self.bilinear_out(
                self.bilinear_proj_sent(cls_sent) * self.bilinear_proj_term(cls_term)
            )
            logits = self.encoder.moe_sk_h(self.dropout(rep))
        
        elif fusion_method == "coattn":
            q_term = out_term.last_hidden_state[:, 0:1, :]
            q_sent = out_sent.last_hidden_state[:, 0:1, :]
            kpm_sent = attention_mask_sent.eq(0)
            kpm_term = attention_mask_term.eq(0)
            
            term_ctx, _ = self.coattn_term_to_sent(q_term, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm_sent)
            sent_ctx, _ = self.coattn_sent_to_term(q_sent, out_term.last_hidden_state, out_term.last_hidden_state, key_padding_mask=kpm_term)
            
            rep = term_ctx.squeeze(1) + sent_ctx.squeeze(1)
            logits = self.encoder.moe_sk_h(self.dropout(rep))
        
        elif fusion_method == "late_interaction":
            sent_tok = out_sent.last_hidden_state
            term_tok = out_term.last_hidden_state
            
            sent_tok = torch.nn.functional.normalize(sent_tok, p=2, dim=-1)
            term_tok = torch.nn.functional.normalize(term_tok, p=2, dim=-1)
            
            sim = torch.matmul(term_tok, sent_tok.transpose(1, 2))
            
            if attention_mask_sent is not None:
                mask = attention_mask_sent.unsqueeze(1).eq(0)
                sim = sim.masked_fill(mask.bool(), torch.finfo(sim.dtype).min)
            
            max_sim = sim.max(dim=-1).values
            
            if attention_mask_term is not None:
                term_valid = attention_mask_term.float()
                denom = term_valid.sum(dim=1).clamp_min(1.0)
                pooled = (max_sim * term_valid).sum(dim=1) / denom
            else:
                pooled = max_sim.mean(dim=1)
            
            cond = self.gate(torch.cat([cls_sent, cls_term], dim=-1))
            rep = cond * pooled.unsqueeze(-1)
            logits = self.encoder.moe_sk_h(self.dropout(rep))
        
        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

        out = self._compute_loss(logits, labels, fusion_method=fusion_method)
        out["moe_stats"] = self._get_moe_stats(fusion_method)
        return out

    def _get_moe_stats(self, fusion_method: str) -> Optional[Dict[str, Any]]:
        moe_module = (
            self.encoder.moe_sk_2h
            if fusion_method == "concat"
            else self.encoder.moe_sk_h
        )
        moe_residual = moe_module.moe_residual
        logits = getattr(moe_residual, "last_router_logits", None)
        if logits is None:
            return None
        return {"router_logits": logits.detach()}
    
    def _collect_aux_loss(self, moe_module: MoESkip) -> torch.Tensor:
        """Collect load-balancing loss from MoE module."""
        moe_residual = moe_module.moe_residual
        logits = getattr(moe_residual, "last_router_logits", None)
        topk_idx = getattr(moe_residual, "last_topk_idx", None)
        
        if logits is None or topk_idx is None:
            return torch.zeros((), device=next(self.parameters()).device)
        
        if logits.ndim != 2 or topk_idx.ndim != 2 or logits.shape[0] == 0:
            return torch.zeros((), device=logits.device)
        
        n_tokens, n_experts = logits.shape
        k = topk_idx.shape[1]
        
        probs = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
        importance = probs.sum(dim=0) / float(n_tokens)
        
        oh = F.one_hot(topk_idx, num_classes=n_experts).to(dtype=probs.dtype)
        load = oh.sum(dim=(0, 1)) / float(n_tokens * k)
        
        aux = n_experts * torch.sum(importance * load)
        aux = torch.clamp(aux, min=0.0, max=10.0)
        return aux
    
    def _aux_weight(self) -> float:
        """Compute auxiliary loss weight with warmup."""
        w = float(self.aux_loss_weight)
        if not self.training:
            return w
        if self.aux_warmup_steps and self.aux_warmup_steps > 0:
            t = min(1.0, float(self._global_step) / float(self.aux_warmup_steps))
            return w * t
        return w
    
    def _jitter(self) -> float:
        """Compute router jitter with warmup."""
        if not self.training:
            return float(self._jitter_start)
        if self.jitter_warmup_steps and self.jitter_warmup_steps > 0:
            t = min(1.0, float(self._global_step) / float(self.jitter_warmup_steps))
            return float(self._jitter_start) * (1.0 - t) + float(self.jitter_end) * t
        return float(self._jitter_start)
    
    def _collect_router_entropy(self, moe_module: MoESkip) -> torch.Tensor:
        """Compute normalized routing entropy."""
        moe_residual = moe_module.moe_residual
        logits = getattr(moe_residual, "last_router_logits", None)
        
        if logits is None or logits.ndim != 2 or logits.shape[0] == 0:
            return torch.zeros((), device=next(self.parameters()).device)
        
        n_experts = logits.shape[-1]
        probs = torch.softmax(logits.float(), dim=-1)
        ent = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)
        ent = ent.mean()
        ent = ent / float(math.log(n_experts + 1e-9))
        return ent.to(dtype=logits.dtype, device=logits.device)
    
    def _compute_loss(self, logits, labels, fusion_method: str = "concat"):
        if labels is None:
            return {
                "loss": None,
                "logits": logits,
                "aux_loss": None,
                "loss_main": None,
                "loss_lambda": None,
                "loss_total": None,
            }
        
        # Compute main classification loss
        if self.loss_type == "ce":
            loss_main = F.cross_entropy(logits, labels)
        elif self.loss_type == "weighted_ce":
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss_main = F.cross_entropy(logits, labels, weight=w)
        elif self.loss_type == "focal":
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss_fn = FocalLoss(gamma=self.focal_gamma, alpha=w, reduction="mean")
            loss_main = loss_fn(logits, labels)
        else:
            raise RuntimeError(f"Unexpected loss_type: {self.loss_type}")
        
        # Select appropriate MoE module based on fusion method
        if fusion_method == "concat":
            moe_module = self.encoder.moe_sk_2h
        else:
            moe_module = self.encoder.moe_sk_h
        
        # Collect auxiliary losses
        aux = self._collect_aux_loss(moe_module)
        aux_w = self._aux_weight()
        loss_lambda = aux_w * aux
        
        entropy = self._collect_router_entropy(moe_module)
        loss_entropy = self.router_entropy_weight * entropy
        
        loss_total = loss_main + loss_lambda + loss_entropy

        
        return {
            "loss": loss_total,
            "logits": logits,
            "aux_loss": aux,
            "router_entropy": entropy,
            "loss_main": loss_main,
            "loss_lambda": loss_lambda,
            "loss_entropy": loss_entropy,
            "loss_total": loss_total,
            "beta_h": self.encoder.moe_sk_h._get_beta(),
            "beta_2h": self.encoder.moe_sk_2h._get_beta(),
        }
    
    @torch.no_grad()
    def _moe_debug_stats(self, moe_module: MoESkip, name: str) -> Optional[Dict[str, Any]]:
        """Debug statistics for MoE skip connection module."""
        def _entropy_from_dist(p: torch.Tensor) -> torch.Tensor:
            eps = 1e-12
            return -(p * (p + eps).log()).sum()
        
        def _cv(p: torch.Tensor) -> float:
            mu = p.mean().clamp_min(1e-12)
            return float((p.std(unbiased=False) / mu).item())
        
        moe_residual = moe_module.moe_residual
        logits = getattr(moe_residual, "last_router_logits", None)
        topk_idx = getattr(moe_residual, "last_topk_idx", None)
        
        if logits is None or topk_idx is None:
            return None
        
        E = int(moe_residual.num_experts)
        if E <= 0:
            return None
        
        probs = torch.softmax(logits, dim=-1)
        
        counts = torch.zeros(E, device=logits.device, dtype=torch.float32)
        counts.scatter_add_(
            0,
            topk_idx.reshape(-1),
            torch.ones_like(topk_idx.reshape(-1), dtype=torch.float32),
        )
        usage_hard = counts / counts.sum().clamp_min(1.0)
        
        usage_soft = probs.mean(dim=0)
        usage_soft = usage_soft / usage_soft.sum().clamp_min(1e-12)
        
        ent_full = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean()
        ent_full_norm = ent_full / math.log(E)
        
        ent_hard = _entropy_from_dist(usage_hard)
        ent_hard_norm = ent_hard / math.log(E)
        
        ent_soft = _entropy_from_dist(usage_soft)
        ent_soft_norm = ent_soft / math.log(E)
        
        logits_std = float(logits.std(unbiased=False).item())
        logits_maxabs = float(logits.abs().max().item())
        
        top2 = torch.topk(logits, k=min(2, E), dim=-1).values
        if top2.size(-1) >= 2:
            gap = float((top2[:, 0] - top2[:, 1]).mean().item())
        else:
            gap = 0.0
        
        max_hard = float(usage_hard.max().item())
        min_hard = float(usage_hard.min().item())
        max_soft = float(usage_soft.max().item())
        min_soft = float(usage_soft.min().item())
        
        return {
            "name": name,
            "H_full_norm": float(ent_full_norm.item()),
            "H_hard_norm": float(ent_hard_norm.item()),
            "H_soft_norm": float(ent_soft_norm.item()),
            "logits_std": logits_std,
            "logits_maxabs": logits_maxabs,
            "gap_top1_top2": gap,
            "usage_hard": usage_hard.detach().cpu(),
            "usage_soft": usage_soft.detach().cpu(),
            "max_hard": max_hard,
            "min_hard": min_hard,
            "max_soft": max_soft,
            "min_soft": min_soft,
            "cv_hard": _cv(usage_hard),
            "cv_soft": _cv(usage_soft),
            "beta": moe_module._get_beta(),
        }
    
    def print_moe_debug(self, topn: int = 3, bottomn: int = 3, eps_dead: float = 1e-6):
        """Print debug statistics for both MoE modules."""
        s_h = self._moe_debug_stats(self.encoder.moe_sk_h, "H")
        s_2h = self._moe_debug_stats(self.encoder.moe_sk_2h, "2H")
        
        if s_h is None and s_2h is None:
            print("[MoE-Skip] No stats yet.")
            return
        
        print("\n[MoE-Skip Debug]")
        
        for s in [s_h, s_2h]:
            if s is None:
                continue
            
            uh = s["usage_hard"].float()
            us = s["usage_soft"].float()
            
            dead_h0 = int((uh == 0).sum().item())
            dead_h = int((uh < eps_dead).sum().item())
            dead_s0 = int((us == 0).sum().item())
            dead_s = int((us < eps_dead).sum().item())
            
            topk = min(topn, uh.numel())
            botk = min(bottomn, uh.numel())
            
            topv_h, topi_h = torch.topk(uh, k=topk, largest=True)
            botv_h, boti_h = torch.topk(uh, k=botk, largest=False)
            topv_s, topi_s = torch.topk(us, k=topk, largest=True)
            botv_s, boti_s = torch.topk(us, k=botk, largest=False)
            
            top_pairs_h = " ".join([f"e{int(i)}={float(v):.6f}" for v, i in zip(topv_h, topi_h)])
            bot_pairs_h = " ".join([f"e{int(i)}={float(v):.6f}" for v, i in zip(botv_h, boti_h)])
            top_pairs_s = " ".join([f"e{int(i)}={float(v):.6f}" for v, i in zip(topv_s, topi_s)])
            bot_pairs_s = " ".join([f"e{int(i)}={float(v):.6f}" for v, i in zip(botv_s, boti_s)])
            
            imb_h = float(s["max_hard"] / (s["min_hard"] + 1e-12))
            imb_s = float(s["max_soft"] / (s["min_soft"] + 1e-12))
            
            print(
                f"MoE-Sk [{s['name']}] beta={s['beta']:.4f} | "
                f"H_full={s['H_full_norm']:.6f} H_soft={s['H_soft_norm']:.6f} H_hard={s['H_hard_norm']:.6f} | "
                f"logits_std={s['logits_std']:.6f} maxabs={s['logits_maxabs']:.6f} gap12={s['gap_top1_top2']:.6f}"
            )
            print(
                f"  HARD: min={s['min_hard']:.6f} max={s['max_hard']:.6f} cv={s['cv_hard']:.3f} imb={imb_h:.2f} "
                f"dead(==0)={dead_h0} dead(<{eps_dead:g})={dead_h}"
            )
            print(f"    top: {top_pairs_h}")
            print(f"    bot: {bot_pairs_h}")
            print(
                f"  SOFT: min={s['min_soft']:.6f} max={s['max_soft']:.6f} cv={s['cv_soft']:.3f} imb={imb_s:.2f} "
                f"dead(==0)={dead_s0} dead(<{eps_dead:g})={dead_s}"
            )
            print(f"    top: {top_pairs_s}")
            print(f"    bot: {bot_pairs_s}")
            print()
    
    def configure_topk_schedule(
        self,
        *,
        enabled: bool,
        start_k: int,
        end_k: int,
        switch_epoch: int,
    ) -> None:
        """Configure top-k schedule for both MoE modules."""
        self._topk_schedule_enabled = bool(enabled)
        self._topk_start = int(start_k)
        self._topk_end = int(end_k)
        self._topk_switch_epoch = int(switch_epoch)
        
        k = self._topk_start if self._topk_schedule_enabled else self._topk_end
        self.encoder.moe_sk_h.set_top_k(k)
        self.encoder.moe_sk_2h.set_top_k(k)
    
    def set_epoch(self, epoch_idx_0based: int) -> None:
        """Update top-k based on epoch if schedule is enabled."""
        if not getattr(self, "_topk_schedule_enabled", False):
            return
        
        if epoch_idx_0based < self._topk_switch_epoch:
            k = self._topk_start
        else:
            k = self._topk_end
        
        self.encoder.moe_sk_h.set_top_k(k)
        self.encoder.moe_sk_2h.set_top_k(k)
        
