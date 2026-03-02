import math
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
import torch.nn as nn

from src.core.loss.focal_loss import FocalLoss
from src.models.base_model import BaseModel


class MoE(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        dropout_p: float,
        act_fn: nn.Module,
        router_bias: bool,
        router_jitter: float,
        capacity_factor: float,
        route_mask_pad_tokens: bool,
        router_temperature: float,
        layer_norm,
    ) -> None:
        super().__init__()
        assert 1 <= top_k <= num_experts

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_experts = int(num_experts)
        self.moe_top_k = int(top_k)
        self.dropout = nn.Dropout(float(dropout_p))
        self.act_fn = act_fn
        self.router_jitter = float(router_jitter)
        self.capacity_factor = capacity_factor
        self.route_mask_pad_tokens = bool(route_mask_pad_tokens)
        self.router_temperature = router_temperature

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=bool(router_bias))

        self.experts_dense1 = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.intermediate_size) for _ in range(self.num_experts)]
        )
        self.experts_dense2 = nn.ModuleList(
            [nn.Linear(self.intermediate_size, self.hidden_size) for _ in range(self.num_experts)]
        )

        self.ln = layer_norm if layer_norm is not None else nn.LayerNorm(self.hidden_size)

        self.last_router_logits = None
        self.last_topk_idx = None

        # init router near-uniform, but break symmetry (important for top-k routing)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)

    def forward(self, hidden_states: torch.Tensor, *, token_mask) -> torch.Tensor:
        bsz, seqlen, hdim = hidden_states.shape
        x = hidden_states.reshape(-1, hdim)  # [N, H]

        active_idx = None
        x_active = x

        if self.route_mask_pad_tokens and token_mask is not None:
            m = token_mask.reshape(-1).bool()
            if not torch.any(m):
                self.last_router_logits = None
                self.last_topk_idx = None
                return self.ln(hidden_states)
            active_idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
            x_active = x.index_select(0, active_idx)

        if self.router_jitter > 0.0:
            noise = (torch.rand_like(x_active) - 0.5) * 2.0 * self.router_jitter
            x_route = x_active + noise
        else:
            x_route = x_active

        router_logits = self.router(x_route) / self.router_temperature  # [N_active, E]
                
        topk_vals, topk_idx = torch.topk(router_logits, k=self.moe_top_k, dim=-1)  # [N_active, K]
        topk_w = torch.softmax(topk_vals, dim=-1)  # [N_active, K]

        # cache for aux loss and debug
        self.last_router_logits = router_logits
        self.last_topk_idx = topk_idx

        cap = None
        if self.capacity_factor is not None:
            cap = int(math.ceil((x_active.shape[0] / self.num_experts) * float(self.capacity_factor)))
            cap = max(cap, 1)

        out_active = torch.zeros_like(x_active)

        flat_idx = topk_idx.reshape(-1)  # [N_active*K]
        flat_tok = torch.arange(x_active.shape[0], device=x_active.device).repeat_interleave(self.moe_top_k)
        flat_kpos = torch.arange(self.moe_top_k, device=x_active.device).repeat(x_active.shape[0])

        for e in range(self.num_experts):
            sel = (flat_idx == e)
            if not torch.any(sel):
                continue

            tok_pos = flat_tok[sel]
            k_pos = flat_kpos[sel]

            if cap is not None and tok_pos.numel() > cap:
                w_sel = topk_w.index_select(0, tok_pos).gather(1, k_pos.unsqueeze(1)).squeeze(1)
                keep = torch.topk(w_sel, k=cap, largest=True, sorted=False).indices
                tok_pos = tok_pos.index_select(0, keep)
                k_pos = k_pos.index_select(0, keep)

            xe = x_active.index_select(0, tok_pos)
            y = self.experts_dense1[e](xe)
            y = self.act_fn(y)
            y = self.experts_dense2[e](y)
            y = self.dropout(y)

            w = topk_w.index_select(0, tok_pos).gather(1, k_pos.unsqueeze(1)).squeeze(1)
            out_active.index_add_(0, tok_pos, y * w.unsqueeze(1))

        if active_idx is not None:
            out = x.clone()
            out.index_copy_(0, active_idx, out_active)
        else:
            out = out_active

        out = out.reshape(bsz, seqlen, hdim)
        return self.ln(out + hidden_states)

    def set_top_k(self, k: int) -> None:
        k = int(k)
        if k < 1:
            k = 1
        if k > self.num_experts:
            k = self.num_experts
        self.moe_top_k = k


class EncoderWithMoEHead(nn.Module):
    """Wrapper that adds a single MoE layer on top of base encoder output"""
    def __init__(self, *, base_encoder: nn.Module, moe_ffn: MoE) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.moe_ffn = moe_ffn

    def forward(self, *args, **kwargs):
        outputs = self.base_encoder(*args, **kwargs)

        attn_mask = kwargs.get("attention_mask", None)
        token_mask = None
        if attn_mask is not None:
            if attn_mask.dim() == 4:
                m = attn_mask[:, 0, 0, :]
                if m.dtype == torch.bool:
                    token_mask = m.long()
                else:
                    m_f = m.float()
                    if torch.min(m_f) < 0.0 and torch.max(m_f) <= 0.0:
                        token_mask = (m_f == 0.0).long()
                    else:
                        token_mask = (m_f > 0.0).long()
            else:
                token_mask = attn_mask

        if isinstance(outputs, (tuple, list)):
            last_hidden = outputs[0]
            new_hidden = self.moe_ffn(last_hidden, token_mask=token_mask)
            return (new_hidden,) + tuple(outputs[1:])

        if hasattr(outputs, "last_hidden_state"):
            new_hidden = self.moe_ffn(outputs.last_hidden_state, token_mask=token_mask)
            outputs.last_hidden_state = new_hidden
            return outputs

        return self.moe_ffn(outputs, token_mask=token_mask)


class MoEHead(BaseModel):
    """Model with a single MoE layer on top of the encoder"""
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
        self.aux_loss_weight = float(moe_cfg["load_balance"].get("coef", 0.0))
        self.router_entropy_weight = float(router_cfg.get("entropy_weight", 0.0))
        self.aux_warmup_steps = int(router_cfg.get("aux_warmup_steps", 0))
        self.jitter_warmup_steps = int(router_cfg.get("jitter_warmup_steps", 0))
        self.jitter_end = float(router_cfg.get("jitter_end", 0.0))
        self._global_step = 0

        cfg = getattr(self.encoder, "config", None)
        hidden_size = int(getattr(cfg, "hidden_size"))
        intermediate_size = int(getattr(cfg, "intermediate_size", hidden_size * 4))
        hidden_act = str(getattr(cfg, "hidden_act", "gelu")).lower()
        act_fn = nn.GELU() if hidden_act == "gelu" else nn.ReLU()
        dropout_p = float(getattr(cfg, "hidden_dropout_prob", model_cfg["encoder"].get("dropout", 0.1)))
        self._jitter_start = float(router_cfg.get("noise_std", 0.0))

        moe_head = MoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=int(moe_cfg["num_experts"]),
            top_k=int(moe_cfg["top_k"]),
            dropout_p=dropout_p,
            act_fn=act_fn,
            router_bias=bool(router_cfg.get("bias", True)),
            router_jitter=float(router_cfg.get("noise_std", 0.0)),
            capacity_factor=moe_cfg.get("capacity_factor"),
            route_mask_pad_tokens=bool(moe_cfg.get("route_mask_pad_tokens", True)),
            router_temperature=float(router_cfg.get("temperature", 1.0)),
            layer_norm=nn.LayerNorm(hidden_size),
        )

        base_encoder = self.encoder
        self.encoder = EncoderWithMoEHead(base_encoder=base_encoder, moe_ffn=moe_head)

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
            moe = getattr(self.encoder, "moe_ffn", None)
            if moe is not None:
                moe.router_jitter = float(self._jitter())

        out = super().forward(
            input_ids_sent=input_ids_sent,
            attention_mask_sent=attention_mask_sent,
            input_ids_term=input_ids_term,
            attention_mask_term=attention_mask_term,
            labels=labels,
            fusion_method=fusion_method,
        )
        moe_stats = self._get_moe_stats()
        if moe_stats is not None:
            out["moe_stats"] = moe_stats
        return out

    def _get_moe_stats(self) -> Optional[Dict[str, Any]]:
        moe = getattr(self.encoder, "moe_ffn", None)
        if moe is None:
            return None
        logits = getattr(moe, "last_router_logits", None)
        if logits is None:
            return None
        return {"router_logits": logits.detach()}

    def _collect_aux_loss(self):
        """Load-balancing loss for single MoE head"""
        moe = getattr(self.encoder, "moe_ffn", None)
        if moe is None:
            return torch.zeros((), device=next(self.parameters()).device)

        logits = getattr(moe, "last_router_logits", None)
        topk_idx = getattr(moe, "last_topk_idx", None)
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
        w = float(self.aux_loss_weight)
        if not self.training:
            return w
        if self.aux_warmup_steps and self.aux_warmup_steps > 0:
            t = min(1.0, float(self._global_step) / float(self.aux_warmup_steps))
            return w * t
        return w

    def _jitter(self) -> float:
        if not self.training:
            return float(self._jitter_start)
        if self.jitter_warmup_steps and self.jitter_warmup_steps > 0:
            t = min(1.0, float(self._global_step) / float(self.jitter_warmup_steps))
            return float(self._jitter_start) * (1.0 - t) + float(self.jitter_end) * t
        return float(self._jitter_start)

    def _collect_router_entropy(self) -> torch.Tensor:
        """Mean per-token routing entropy for single MoE head"""
        moe = getattr(self.encoder, "moe_ffn", None)
        if moe is None:
            return torch.zeros((), device=next(self.parameters()).device)

        logits = getattr(moe, "last_router_logits", None)
        if logits is None or logits.ndim != 2 or logits.shape[0] == 0:
            return torch.zeros((), device=next(self.parameters()).device)

        n_experts = logits.shape[-1]
        probs = torch.softmax(logits.float(), dim=-1)
        ent = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)
        ent = ent.mean()
        ent = ent / float(math.log(n_experts + 1e-9))
        return ent.to(dtype=logits.dtype, device=logits.device)

    def _compute_loss(self, logits, labels):
        if labels is None:
            return {
                "loss": None,
                "logits": logits,
                "aux_loss": None,
                "loss_main": None,
                "loss_lambda": None,
                "loss_total": None,
            }

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
        
        aux = self._collect_aux_loss()
        aux_w = self._aux_weight()
        loss_lambda = aux_w * aux

        entropy = self._collect_router_entropy()
        loss_entropy = float(self.router_entropy_weight) * entropy

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
        }

    @torch.no_grad()
    def _moe_debug_stats(self):
        """Debug statistics for single MoE head"""
        def _entropy_from_dist(p: torch.Tensor) -> torch.Tensor:
            eps = 1e-12
            return -(p * (p + eps).log()).sum()

        def _cv(p: torch.Tensor) -> float:
            mu = p.mean().clamp_min(1e-12)
            return float((p.std(unbiased=False) / mu).item())

        moe = getattr(self.encoder, "moe_ffn", None)
        if moe is None:
            return None

        logits = getattr(moe, "last_router_logits", None)
        topk_idx = getattr(moe, "last_topk_idx", None)
        if logits is None or topk_idx is None:
            return None

        E = int(getattr(moe, "num_experts", 0) or 0)
        if E <= 0:
            return None

        logits2 = logits.reshape(-1, logits.size(-1)) if logits.dim() == 3 else logits
        topk2 = topk_idx.reshape(-1, topk_idx.size(-1)) if topk_idx.dim() == 3 else topk_idx

        probs = torch.softmax(logits2, dim=-1)

        counts = torch.zeros(E, device=logits2.device, dtype=torch.float32)
        counts.scatter_add_(
            0,
            topk2.reshape(-1),
            torch.ones_like(topk2.reshape(-1), dtype=torch.float32),
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

        logits_std = float(logits2.std(unbiased=False).item())
        logits_maxabs = float(logits2.abs().max().item())

        top2 = torch.topk(logits2, k=min(2, E), dim=-1).values
        if top2.size(-1) >= 2:
            gap = float((top2[:, 0] - top2[:, 1]).mean().item())
        else:
            gap = 0.0

        max_hard = float(usage_hard.max().item())
        min_hard = float(usage_hard.min().item())
        max_soft = float(usage_soft.max().item())
        min_soft = float(usage_soft.min().item())

        return {
            "layer": -1,
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
        }

    def print_moe_debug(self, topn: int = 3, bottomn: int = 3, eps_dead: float = 1e-6):
        s = self._moe_debug_stats()
        if s is None:
            print("[MoE] No stats yet.")
            return

        print("\n[MoE Debug - Single Head]")
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
            f"MoE Head | "
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
        self._topk_schedule_enabled = bool(enabled)
        self._topk_start = int(start_k)
        self._topk_end = int(end_k)
        self._topk_switch_epoch = int(switch_epoch)

        if self._topk_schedule_enabled:
            self.encoder.moe_ffn.set_top_k(self._topk_start)
        else:
            self.encoder.moe_ffn.set_top_k(self._topk_end)

    def set_epoch(self, epoch_idx_0based: int) -> None:
        if not getattr(self, "_topk_schedule_enabled", False):
            return

        if epoch_idx_0based < self._topk_switch_epoch:
            k = self._topk_start
        else:
            k = self._topk_end

        self.encoder.moe_ffn.set_top_k(k)
