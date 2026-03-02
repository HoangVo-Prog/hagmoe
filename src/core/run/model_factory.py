"""
Model factory for decoupled model construction.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict
import copy

from src.core.config import validate_config
from src.core.utils.const import DEVICE


_MODEL_CLASS_MAP: Dict[str, str] = {
    "BaseModel": "BaseModel",
    "BERTSPCModel": "BERTSPCModel",
    "MoEFFN": "MoEFFN",
    "MoEHead": "MoEHead",
    "MoESkip": "MoESkipModel",
    "HAGMoE": "HAGMoE",
}

_MODEL_ALIASES: Dict[str, str] = {
    "MoESkipModel": "MoESkip",
}


def _resolve_model_name(cfg: Any) -> str:
    if hasattr(cfg, "_schema_cfg"):
        name = str(cfg._schema_cfg.get("model", {}).get("name") or "").strip()
        if name:
            return name
    if isinstance(cfg, dict):
        name = str(cfg.get("model", {}).get("name") or cfg.get("mode") or "").strip()
        if name:
            return name
    return str(getattr(cfg, "mode", "") or "").strip()


def build_model(cfg) -> Any:
    """
    Build a model instance from config.
    Ensures validate_config(cfg) has been called.
    """
    validate_config(cfg)

    model_name = _resolve_model_name(cfg)
    if model_name in _MODEL_ALIASES:
        model_name = _MODEL_ALIASES[model_name]

    if model_name not in _MODEL_CLASS_MAP:
        known = sorted(_MODEL_CLASS_MAP.keys())
        raise ValueError(f"Unknown model name '{model_name}'. Expected one of: {known}")

    models_mod = import_module("src.models")
    class_name = _MODEL_CLASS_MAP[model_name]
    if not hasattr(models_mod, class_name):
        raise ImportError(f"Model class '{class_name}' not found in src.models")

    ModelCls = getattr(models_mod, class_name)

    schema_cfg = getattr(cfg, "_schema_cfg", None)
    if schema_cfg is None:
        schema_cfg = validate_config(cfg)._schema_cfg

    model_cfg = copy.deepcopy(schema_cfg.get("model", {}))
    moe_cfg = copy.deepcopy(schema_cfg.get("moe", {}))
    loss_cfg = copy.deepcopy(schema_cfg.get("loss", {}))

    # Attach labels for HAGMoE (explicit wiring)
    labels_cfg = model_cfg.setdefault("labels", {})
    labels_cfg.setdefault("id2label", getattr(cfg, "id2label", None))
    labels_cfg.setdefault("label2id", getattr(cfg, "label2id", None))

    encoder_ref = model_cfg.get("encoder", {}).get("pretrained_name")

    return ModelCls(
        encoder=encoder_ref,
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        moe_cfg=moe_cfg,
    ).to(DEVICE)
