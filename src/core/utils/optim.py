from __future__ import annotations

from typing import Iterable, Optional, List, Dict

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def _is_no_decay(name: str) -> bool:
    name = name or ""
    if name.endswith(".bias") or name.endswith("bias"):
        return True
    # Common LayerNorm patterns
    if "LayerNorm.weight" in name or "LayerNorm.bias" in name:
        return True
    if "layer_norm.weight" in name or "layer_norm.bias" in name:
        return True
    return False


def build_optimizer_and_scheduler(
    *,
    model,
    lr: float,
    warmup_ratio: float,
    total_steps: int,
    # New: head lr for discriminative training (Schedule A)
    lr_head: Optional[float] = None,
    weight_decay: float = 0.01,
    params: Optional[Iterable] = None,
    adamw_foreach: bool = False,
    adamw_fused: bool = False,
):
    """
    AdamW builder with optional discriminative LR for encoder vs head.

    - Encoder params (under `model.encoder`) use `lr`
    - Head params (everything else) use `lr_head` (defaults to `lr`)
    - Applies weight decay to non-bias / non-LayerNorm params only.
    """
    if lr_head is None:
        lr_head = lr

    # Restrict to a provided param subset (e.g., requires_grad-filtered list) if given.
    allowed_ids = None
    if params is not None:
        allowed_ids = {id(p) for p in params}

    enc_decay: List = []
    enc_no_decay: List = []
    head_decay: List = []
    head_no_decay: List = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if allowed_ids is not None and id(p) not in allowed_ids:
            continue

        in_encoder = name.startswith("encoder.")
        no_decay = _is_no_decay(name)

        if in_encoder:
            (enc_no_decay if no_decay else enc_decay).append(p)
        else:
            (head_no_decay if no_decay else head_decay).append(p)

    param_groups: List[Dict] = []
    if enc_decay:
        param_groups.append({"params": enc_decay, "lr": lr, "weight_decay": weight_decay})
    if enc_no_decay:
        param_groups.append({"params": enc_no_decay, "lr": lr, "weight_decay": 0.0})
    if head_decay:
        param_groups.append({"params": head_decay, "lr": lr_head, "weight_decay": weight_decay})
    if head_no_decay:
        param_groups.append({"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0})

    # Fallback: if for some reason grouping missed everything, just use provided params or model params.
    if not param_groups:
        base_params = list(params) if params is not None else list(model.parameters())
        param_groups = [{"params": base_params, "lr": lr, "weight_decay": weight_decay}]

    try:
        optimizer = AdamW(
            param_groups,
            foreach=adamw_foreach,
            fused=adamw_fused,
        )
    except TypeError:
        optimizer = AdamW(
            param_groups,
            foreach=adamw_foreach,
        )

    warmup_steps = int(float(warmup_ratio) * int(total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=int(total_steps),
    )
    return optimizer, scheduler
