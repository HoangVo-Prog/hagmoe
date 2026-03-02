import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from src.models.base_model import BaseModel
from src.core.loss.focal_loss import FocalLoss


class MoE(nn.Module):
    """
    Replaces (Intermediate + Output) FFN with MoE FFN.
    Optionally masks pad tokens from routing via attention_mask.
    Stores last router info for aux loss.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_p: float,
        layer_norm_eps: float,
        act_fn,
        base_dense1: nn.Linear,
        base_dense2: nn.Linear,
        base_layernorm: nn.LayerNorm,
        moe_top_k: int,
        num_experts:int,
        router_bias: bool,
        route_mask_pad_tokens: bool,
        capacity_factor: float,
    ):
        super().__init__()
        assert 1 <= moe_top_k <= num_experts

        self.moe_top_k = moe_top_k
        self.num_experts = num_experts
        self.route_bias = router_bias
        self.route_mask_pad_tokens = route_mask_pad_tokens
        self.capacity_factor = capacity_factor
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = nn.Dropout(dropout_p)
        self.act_fn = act_fn

        self.router = nn.Linear(hidden_size, num_experts, bias=router_bias)

        self.expert_dense1 = nn.ModuleList(
            [nn.Linear(hidden_size, intermediate_size) for _ in range(num_experts)]
        )
        self.expert_dense2 = nn.ModuleList(
            [nn.Linear(intermediate_size, hidden_size) for _ in range(num_experts)]
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # init experts from original FFN weights
        for e in range(num_experts):
            self.expert_dense1[e].weight.data.copy_(base_dense1.weight.data)
            self.expert_dense1[e].bias.data.copy_(base_dense1.bias.data)
            self.expert_dense2[e].weight.data.copy_(base_dense2.weight.data)
            self.expert_dense2[e].bias.data.copy_(base_dense2.bias.data)

        self.layer_norm.weight.data.copy_(base_layernorm.weight.data)
        self.layer_norm.bias.data.copy_(base_layernorm.bias.data)

        # init router near-uniform
        nn.init.zeros_(self.router.weight)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)

        # cache for aux loss
        self.last_router_logit = None
        self.last_topk_idx = None

    @torch.no_grad()
    def _apply_capacity(self, token_idx: torch.Tensor, max_tokens: int) -> torch.Tensor:
        if token_idx.numel() <= max_tokens:
            return token_idx
        return token_idx[:max_tokens]

    def forward(self, hidden_states: torch.Tensor, token_mask = None) -> torch.Tensor:
        bsz, seqlen, h = hidden_states.shape
        x = hidden_states
        flat = x.reshape(-1, h)  # [N, H]
        n_tokens = flat.size(0)

        logits = self.router(flat)  # [N, E]
        active_idx = None
        if self.route_mask_pad_tokens and token_mask is not None:
            # token_mask: [B,T] with 1 for real tokens, 0 for pad
            mask_flat = token_mask.reshape(-1).to(dtype=torch.bool)  # [N]
            active_idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(-1)  # [Na]

            # Nếu batch hiếm khi toàn pad (Na == 0) thì bypass MoE, trả về residual+LN
            if active_idx.numel() == 0:
                out = self.layer_norm(hidden_states + 0.0)
                # cache empty để aux loss không crash
                self.last_router_logits = logits[:0]
                self.last_topk_idx = torch.empty((0, self.moe_top_k), device=logits.device, dtype=torch.long)
                return out

            logits_active = logits.index_select(0, active_idx)  # [Na, E]
        else:
            logits_active = logits  # [N, E]

        topk_vals, topk_idx = torch.topk(logits_active, k=self.moe_top_k, dim=-1)  # [Na, K] or [N, K]
        topk_w = F.softmax(topk_vals, dim=-1)

        # cache for aux loss (only active tokens)
        self.last_router_logits = logits_active
        self.last_topk_idx = topk_idx

        out_active = flat.new_zeros((logits_active.size(0), h))  # [Na, H] or [N, H]

        max_tokens_per_expert = None
        if self.capacity_factor is not None:
            nt = logits_active.size(0)
            max_tokens_per_expert = int(((nt / self.num_experts) * self.capacity_factor) + 0.999)

        for e in range(self.num_experts):
            mask = topk_idx.eq(e)
            if not mask.any():
                continue

            tok_pos, k_pos = torch.where(mask)
            if max_tokens_per_expert is not None:
                tok_pos = self._apply_capacity(tok_pos, max_tokens_per_expert)
                k_pos = k_pos[: tok_pos.numel()]

            if active_idx is not None:
                flat_active = flat.index_select(0, active_idx)  # [Na,H]
                x_e = flat_active.index_select(0, tok_pos)      # [M,H]
            else:
                x_e = flat.index_select(0, tok_pos)
            w_e = topk_w[tok_pos, k_pos].unsqueeze(-1)

            y = self.expert_dense1[e](x_e)
            y = self.act_fn(y)
            y = self.expert_dense2[e](y)
            y = self.dropout(y)

            out_active.index_add_(0, tok_pos, y * w_e)

        # scatter back if masked routing
        out = flat.new_zeros((n_tokens, h))
        if active_idx is not None:
            out.index_copy_(0, active_idx, out_active)
        else:
            out = out_active

        out = out.view(bsz, seqlen, h)
        out = self.layer_norm(out + x)
        return out


def _get_act_fn_from_intermediate(intermediate_module: nn.Module):
    if hasattr(intermediate_module, "intermediate_act_fn"):
        return intermediate_module.intermediate_act_fn
    if hasattr(intermediate_module, "activation"):
        return intermediate_module.activation
    raise ValueError("Cannot find activation function on intermediate module")


def replace_encoder_ffn_with_moe(
    encoder: nn.Module,
    moe_top_k: int,
    num_experts:int,
    router_bias: bool,
    route_mask_pad_tokens: bool,
    capacity_factor: float,
) -> None:
    if not hasattr(encoder, "encoder") or not hasattr(encoder.encoder, "layer"):
        raise ValueError("Encoder does not look like BERT/RoBERTa model with encoder.layer")

    for layer in encoder.encoder.layer:
        intermediate = layer.intermediate
        output = layer.output

        base_dense1 = intermediate.dense
        base_dense2 = output.dense
        base_ln = output.LayerNorm
        dropout_p = output.dropout.p if hasattr(output.dropout, "p") else encoder.config.hidden_dropout_prob
        act_fn = _get_act_fn_from_intermediate(intermediate)

        moe = MoE(
            hidden_size=encoder.config.hidden_size,
            intermediate_size=encoder.config.intermediate_size,
            dropout_p=dropout_p,
            layer_norm_eps=encoder.config.layer_norm_eps,
            act_fn=act_fn,
            base_dense1=base_dense1,
            base_dense2=base_dense2,
            base_layernorm=base_ln,
            moe_top_k=moe_top_k,
            num_experts=num_experts,
            router_bias=router_bias,
            route_mask_pad_tokens=route_mask_pad_tokens,
            capacity_factor=capacity_factor,
        )

        layer.intermediate = nn.Identity()
        layer.output = nn.Identity()
        layer.moe_ffn = moe

        def new_forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            **kwargs,
        ):
            past = kwargs.get("past_key_values", None)
            if past is None:
                past = kwargs.get("past_key_value", None)

            try:
                self_attention_outputs = self.attention(
                    hidden_states,
                    attention_mask,
                    head_mask,
                    output_attentions=output_attentions,
                    past_key_values=past,
                )
            except TypeError:
                self_attention_outputs = self.attention(
                    hidden_states,
                    attention_mask,
                    head_mask,
                    output_attentions=output_attentions,
                    past_key_value=past,
                )
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:]

            if encoder_hidden_states is not None:
                cross_attention_outputs = self.crossattention(
                    attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[1:]

            # token_mask is 2D for route_mask_pad_tokens, do not use extended 4D mask
            token_mask = None
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    token_mask = (attention_mask[:, 0, 0, :] == 0).to(dtype=torch.long)
                elif attention_mask.dim() == 2:
                    token_mask = attention_mask.to(dtype=torch.long)

            layer_output = self.moe_ffn(attention_output, token_mask=token_mask)
            return (layer_output,) + outputs

        layer.forward = new_forward.__get__(layer, layer.__class__)


def moe_load_balance_loss(
    router_logits: torch.Tensor,  # [N, E]
    topk_idx: torch.Tensor,       # [N, K]
    num_experts: int,
) -> torch.Tensor:
    probs = torch.softmax(router_logits, dim=-1)  # [N, E]
    importance = probs.mean(dim=0)                # [E]

    # load from hard top-1 routing
    one_hot = torch.zeros((topk_idx.size(0), num_experts), device=topk_idx.device, dtype=probs.dtype)
    one_hot.scatter_(1, topk_idx[:, :1], 1.0)
    load = one_hot.mean(dim=0)                    # [E]

    return num_experts * torch.sum(importance * load)


class MoEFFN(BaseModel):
    def __init__(
        self,
        *,
        encoder=None,
        model_cfg: dict,
        loss_cfg: dict,
        moe_cfg: dict,
    ) -> None:
        super().__init__(encoder=encoder, model_cfg=model_cfg, loss_cfg=loss_cfg, moe_cfg=moe_cfg)

        self.aux_loss_weight = float(moe_cfg["load_balance"].get("coef", 0.0))

        router_cfg = moe_cfg.get("router", {})
        replace_encoder_ffn_with_moe(
            self.encoder,
            moe_top_k=int(moe_cfg["top_k"]),
            num_experts=int(moe_cfg["num_experts"]),
            router_bias=bool(router_cfg.get("bias", True)),
            route_mask_pad_tokens=bool(moe_cfg.get("route_mask_pad_tokens", True)),
            capacity_factor=moe_cfg.get("capacity_factor"),
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
        out = super().forward(
            input_ids_sent=input_ids_sent,
            attention_mask_sent=attention_mask_sent,
            input_ids_term=input_ids_term,
            attention_mask_term=attention_mask_term,
            labels=labels,
            fusion_method=fusion_method,
        )
        out["moe_stats"] = self._get_moe_stats()
        return out

    def _get_moe_stats(self):
        logits = None
        for layer in self.encoder.encoder.layer:
            moe = getattr(layer, "moe_ffn", None)
            if moe is None:
                continue
            logits = getattr(moe, "last_router_logits", None)
        if logits is None:
            return None
        return {"router_logits": logits.detach()}


    def _collect_aux_loss(self):
        total, count = 0.0, 0
        for layer in self.encoder.encoder.layer:
            moe = getattr(layer, "moe_ffn", None)
            if moe is None or moe.last_router_logits is None:
                continue
            total += moe_load_balance_loss(
                moe.last_router_logits, moe.last_topk_idx, moe.num_experts
            )
            count += 1
        return total / count if count > 0 else torch.tensor(0.0, device=self.device)


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
        loss_lambda = self.aux_loss_weight * aux

        loss_total = loss_main + loss_lambda

        return {
            "loss": loss_total,          
            "logits": logits,            
            "aux_loss": aux,            
            "loss_main": loss_main,      
            "loss_lambda": loss_lambda,  
            "loss_total": loss_total,    
        }
        
    @torch.no_grad()
    def _moe_debug_stats_per_layer(self):
        """Collect debug statistics from all MoE FFN layers"""
        def _entropy_from_dist(p: torch.Tensor) -> torch.Tensor:
            eps = 1e-12
            return -(p * (p + eps).log()).sum()

        def _cv(p: torch.Tensor) -> float:
            mu = p.mean().clamp_min(1e-12)
            return float((p.std(unbiased=False) / mu).item())

        stats = []

        for li, layer in enumerate(self.encoder.encoder.layer):
            moe = getattr(layer, "moe_ffn", None)
            if moe is None:
                continue
            
            logits = getattr(moe, "last_router_logits", None)
            topk_idx = getattr(moe, "last_topk_idx", None)
            if logits is None or topk_idx is None:
                continue

            E = int(getattr(moe, "num_experts", 0) or 0)
            if E <= 0:
                continue

            # logits: [B, T, E] or [N, E]
            logits2 = logits.reshape(-1, logits.size(-1)) if logits.dim() == 3 else logits
            # topk_idx: [B, T, K] or [N, K]
            topk2 = topk_idx.reshape(-1, topk_idx.size(-1)) if topk_idx.dim() == 3 else topk_idx

            # Soft distribution per token
            probs = torch.softmax(logits2, dim=-1)  # [N, E]

            # 1) Hard usage from topk
            counts = torch.zeros(E, device=logits2.device, dtype=torch.float32)
            counts.scatter_add_(
                0,
                topk2.reshape(-1),
                torch.ones_like(topk2.reshape(-1), dtype=torch.float32),
            )
            usage_hard = counts / counts.sum().clamp_min(1.0)

            # 2) Soft expected usage from probs
            usage_soft = probs.mean(dim=0)  # [E]
            usage_soft = usage_soft / usage_soft.sum().clamp_min(1e-12)

            # Entropy
            ent_full = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean()  # scalar
            ent_full_norm = ent_full / math.log(E)

            ent_hard = _entropy_from_dist(usage_hard)
            ent_hard_norm = ent_hard / math.log(E)

            ent_soft = _entropy_from_dist(usage_soft)
            ent_soft_norm = ent_soft / math.log(E)

            # Logits stats
            logits_std = float(logits2.std(unbiased=False).item())
            logits_maxabs = float(logits2.abs().max().item())

            # Margin top1-top2 (how sharp routing is)
            top2 = torch.topk(logits2, k=min(2, E), dim=-1).values  # [N,2]
            if top2.size(-1) >= 2:
                gap = float((top2[:, 0] - top2[:, 1]).mean().item())
            else:
                gap = 0.0

            # Loads
            max_hard = float(usage_hard.max().item())
            min_hard = float(usage_hard.min().item())
            max_soft = float(usage_soft.max().item())
            min_soft = float(usage_soft.min().item())

            stats.append({
                "layer": int(li),
                
                # entropy
                "H_full_norm": float(ent_full_norm.item()),
                "H_hard_norm": float(ent_hard_norm.item()),
                "H_soft_norm": float(ent_soft_norm.item()),

                # logits
                "logits_std": logits_std,
                "logits_maxabs": logits_maxabs,
                "gap_top1_top2": gap,

                # usage
                "usage_hard": usage_hard.detach().cpu(),
                "usage_soft": usage_soft.detach().cpu(),

                # min max
                "max_hard": max_hard,
                "min_hard": min_hard,
                "max_soft": max_soft,
                "min_soft": min_soft,

                # dispersion
                "cv_hard": _cv(usage_hard),
                "cv_soft": _cv(usage_soft),
            })

        return stats

    def print_moe_debug(self, topn: int = 3, bottomn: int = 3, eps_dead: float = 1e-6):
        """Print debug statistics for all MoE FFN layers"""
        stats = self._moe_debug_stats_per_layer()
        if not stats:
            print("[MoE] No stats yet.")
            return

        print("\n[MoE Debug]")
        for s in stats:
            layer_id = int(s["layer"])
            layer_txt = f"{layer_id:02d}"

            uh = s["usage_hard"].float()
            us = s["usage_soft"].float()

            # dead counts
            dead_h0 = int((uh == 0).sum().item())
            dead_h = int((uh < eps_dead).sum().item())
            dead_s0 = int((us == 0).sum().item())
            dead_s = int((us < eps_dead).sum().item())

            # top/bottom
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

            # imbalance
            imb_h = float(s["max_hard"] / (s["min_hard"] + 1e-12))
            imb_s = float(s["max_soft"] / (s["min_soft"] + 1e-12))

            print(
                f"Layer {layer_txt} | "
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
