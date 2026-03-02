"""
Configuration validation and legacy-key mapping.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import warnings
from typing import Any, Dict, Tuple

from .schema import MODEL_NAMES


def _as_dict(cfg: Any) -> dict:
    if isinstance(cfg, dict):
        return dict(cfg)
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)
    return dict(getattr(cfg, "__dict__", {}))


def _warn(msg: str) -> None:
    warnings.warn(msg, category=UserWarning, stacklevel=2)


def _ensure_dict(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in parent or parent[key] is None:
        parent[key] = {}
    if not isinstance(parent[key], dict):
        raise ValueError(f"Config key '{key}' must be a dict.")
    return parent[key]


def _map_legacy_keys(cfg: Any) -> Dict[str, Any]:
    """
    Map flat legacy keys to the canonical nested structure (best-effort).
    Returns a nested dict representation used for validation only.
    """
    flat = _as_dict(cfg)
    schema_cfg: Dict[str, Any] = {}

    model = _ensure_dict(schema_cfg, "model")
    moe = _ensure_dict(schema_cfg, "moe")
    training = _ensure_dict(schema_cfg, "training")
    data = _ensure_dict(schema_cfg, "data")
    loss = _ensure_dict(schema_cfg, "loss")
    optim = _ensure_dict(schema_cfg, "optim")

    # If config is already nested, seed from it first.
    if isinstance(flat.get("model"), dict):
        model.update(flat.get("model", {}))
    if isinstance(flat.get("moe"), dict):
        moe.update(flat.get("moe", {}))
    if isinstance(flat.get("training"), dict):
        training.update(flat.get("training", {}))
    if isinstance(flat.get("data"), dict):
        data.update(flat.get("data", {}))
    if isinstance(flat.get("loss"), dict):
        loss.update(flat.get("loss", {}))
    if isinstance(flat.get("optim"), dict):
        optim.update(flat.get("optim", {}))

    # Model name
    if "mode" in flat:
        mode_val = str(flat.get("mode") or "").strip()
        if mode_val == "MoESkipModel":
            model["name"] = "MoESkip"
            _warn("Legacy key 'mode' MoESkipModel mapped to cfg.model.name=MoESkip")
        else:
            model["name"] = flat.get("mode")
            _warn("Legacy key 'mode' mapped to cfg.model.name")

    # Encoder config
    encoder = _ensure_dict(model, "encoder")
    if "model_name" in flat:
        encoder["pretrained_name"] = flat.get("model_name")
        _warn("Legacy key 'model_name' mapped to cfg.model.encoder.pretrained_name")
    if "max_len_sent" in flat:
        encoder["max_len"] = flat.get("max_len_sent")
        _warn("Legacy key 'max_len_sent' mapped to cfg.model.encoder.max_len")
    if "dropout" in flat:
        encoder["dropout"] = flat.get("dropout")
        _warn("Legacy key 'dropout' mapped to cfg.model.encoder.dropout")

    # Common model fields
    common = _ensure_dict(model, "common")
    if "num_labels" in flat:
        common["num_labels"] = flat.get("num_labels")
        _warn("Legacy key 'num_labels' mapped to cfg.model.common.num_labels")

    # Fusion
    fusion = _ensure_dict(model, "fusion")
    if "fusion_method" in flat:
        fusion["type"] = flat.get("fusion_method")
        _warn("Legacy key 'fusion_method' mapped to cfg.model.fusion.type")
    if "dropout" in flat:
        fusion.setdefault("dropout", flat.get("dropout"))

    # Head
    head = _ensure_dict(model, "head")
    if "head_type" in flat:
        head["type"] = flat.get("head_type")
        _warn("Legacy key 'head_type' mapped to cfg.model.head.type")
    if "type" not in head:
        head["type"] = "linear"

    # BERT-SPC (legacy defaults)
    bert_spc = _ensure_dict(model, "bert_spc")
    bert_spc.setdefault("spc_type", "default")
    bert_spc.setdefault("use_segment_ids", True)

    # Loss
    if "loss_type" in flat:
        loss["type"] = flat.get("loss_type")
        _warn("Legacy key 'loss_type' mapped to cfg.loss.type")
    if "class_weights" in flat:
        loss["class_weights"] = flat.get("class_weights")
        _warn("Legacy key 'class_weights' mapped to cfg.loss.class_weights")
    if "focal_gamma" in flat:
        loss["focal_gamma"] = flat.get("focal_gamma")
        _warn("Legacy key 'focal_gamma' mapped to cfg.loss.focal_gamma")

    # Data
    if "train_path" in flat:
        data["train_path"] = flat.get("train_path")
    if "test_path" in flat:
        data["test_path"] = flat.get("test_path")
    if "max_len_sent" in flat:
        data["max_len_sent"] = flat.get("max_len_sent")
    if "max_len_term" in flat:
        data["max_len_term"] = flat.get("max_len_term")
    if "val_path" in flat:
        data["val_path"] = flat.get("val_path")

    # Training
    if "epochs" in flat:
        training["epochs"] = flat.get("epochs")
    if "train_batch_size" in flat:
        training["batch_size"] = flat.get("train_batch_size")
        _warn("Legacy key 'train_batch_size' mapped to cfg.training.batch_size")
    if "eval_batch_size" in flat:
        training["eval_batch_size"] = flat.get("eval_batch_size")
    if "test_batch_size" in flat:
        training["test_batch_size"] = flat.get("test_batch_size")
    if "freeze_epochs" in flat:
        training["freeze_epochs"] = flat.get("freeze_epochs")
    if "early_stop_patience" in flat:
        training["early_stop_patience"] = flat.get("early_stop_patience")
    if "seed" in flat:
        training["seed"] = flat.get("seed")

    # Optim
    if "lr" in flat:
        optim["lr"] = flat.get("lr")
    if "lr_head" in flat:
        optim["lr_head"] = flat.get("lr_head")
    if "warmup_ratio" in flat:
        optim["warmup_ratio"] = flat.get("warmup_ratio")
    if "weight_decay" in flat:
        optim["weight_decay"] = flat.get("weight_decay")
    if "adamw_foreach" in flat:
        optim["adamw_foreach"] = flat.get("adamw_foreach")
    if "adamw_fused" in flat:
        optim["adamw_fused"] = flat.get("adamw_fused")

    # MoE (legacy flat to moe.*)
    if "num_experts" in flat:
        moe["num_experts"] = flat.get("num_experts")
    if "moe_top_k" in flat:
        moe["top_k"] = flat.get("moe_top_k")
    if "capacity_factor" in flat:
        moe["capacity_factor"] = flat.get("capacity_factor")

    router = _ensure_dict(moe, "router")
    if "router_temperature" in flat:
        router["temperature"] = flat.get("router_temperature")
    if "router_jitter" in flat:
        router["noise_std"] = flat.get("router_jitter")
    if "router_bias" in flat:
        router["bias"] = flat.get("router_bias")
    if "router_entropy_weight" in flat:
        router["entropy_weight"] = flat.get("router_entropy_weight")
    if "router_entropy_target" in flat:
        router["entropy_target"] = flat.get("router_entropy_target")
    if "router_collapse_weight" in flat:
        router["collapse_weight"] = flat.get("router_collapse_weight")
    if "router_collapse_tau" in flat:
        router["collapse_tau"] = flat.get("router_collapse_tau")
    if "aux_warmup_steps" in flat:
        router["aux_warmup_steps"] = flat.get("aux_warmup_steps")
    if "jitter_warmup_steps" in flat:
        router["jitter_warmup_steps"] = flat.get("jitter_warmup_steps")
    if "jitter_end" in flat:
        router["jitter_end"] = flat.get("jitter_end")

    load_balance = _ensure_dict(moe, "load_balance")
    if "aux_loss_weight" in flat:
        load_balance["coef"] = flat.get("aux_loss_weight")
    if "route_mask_pad_tokens" in flat:
        moe["route_mask_pad_tokens"] = flat.get("route_mask_pad_tokens")

    expert = _ensure_dict(moe, "expert")
    if "dropout" in flat:
        expert["dropout"] = flat.get("dropout")

    # MoESkip legacy keys
    sk = _ensure_dict(model, "sk")
    if "beta_start" in flat:
        sk["beta_start"] = flat.get("beta_start")
    if "beta_end" in flat:
        sk["beta_end"] = flat.get("beta_end")
    if "beta_warmup_steps" in flat:
        sk["beta_warmup_steps"] = flat.get("beta_warmup_steps")
    if "expert_hidden" in flat:
        sk["expert_hidden"] = flat.get("expert_hidden")

    # HAGMoE legacy keys
    hag = _ensure_dict(model, "hagmoe")
    if "hag_num_groups" in flat:
        hag["num_groups"] = flat.get("hag_num_groups")
    if "hag_experts_per_group" in flat:
        hag["experts_per_group"] = flat.get("hag_experts_per_group")
    if "hag_router_temperature" in flat:
        hag["router_temperature"] = flat.get("hag_router_temperature")
    if "hag_router_topk_groups" in flat:
        hag["router_topk_groups"] = flat.get("hag_router_topk_groups")
    if "hag_group_temperature" in flat:
        hag["group_temperature"] = flat.get("hag_group_temperature")
    if "hag_group_temperature_anneal" in flat:
        hag["group_temperature_anneal"] = flat.get("hag_group_temperature_anneal")
    if "hag_merge" in flat:
        hag["merge"] = flat.get("hag_merge")
    if "hag_fusion_method" in flat:
        hag["fusion_method"] = flat.get("hag_fusion_method")
    if "hag_use_group_loss" in flat:
        hag["use_group_loss"] = flat.get("hag_use_group_loss")
    if "hag_use_balance_loss" in flat:
        hag["use_balance_loss"] = flat.get("hag_use_balance_loss")
    if "hag_use_diversity_loss" in flat:
        hag["use_diversity_loss"] = flat.get("hag_use_diversity_loss")
    if "hag_lambda_group" in flat:
        hag["lambda_group"] = flat.get("hag_lambda_group")
    if "hag_lambda_balance" in flat:
        hag["lambda_balance"] = flat.get("hag_lambda_balance")
    if "hag_lambda_diversity" in flat:
        hag["lambda_diversity"] = flat.get("hag_lambda_diversity")
    if "hag_verbose_loss" in flat:
        hag["verbose_loss"] = flat.get("hag_verbose_loss")

    # Labels mapping (optional)
    labels = _ensure_dict(model, "labels")
    if "id2label" in flat:
        labels["id2label"] = flat.get("id2label")
    if "label2id" in flat:
        labels["label2id"] = flat.get("label2id")

    # Ensure moe.enabled exists in schema config (or infer for legacy mode)
    name = str(model.get("name") or "").strip()
    if "enabled" not in moe:
        moe["enabled"] = name in {"MoEFFN", "MoEHead", "MoESkip", "HAGMoE"}
        _warn("Legacy config: inferred cfg.moe.enabled from model name")
    elif moe.get("enabled") is False and name in {"MoEFFN", "MoEHead", "MoESkip", "HAGMoE"}:
        moe["enabled"] = True
        _warn("Legacy config: forcing cfg.moe.enabled=True for MoE model")

    return schema_cfg


def _require_keys(cfg: Dict[str, Any], path: str, keys: Tuple[str, ...]) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required keys at '{path}': {joined}")


def validate_base_model(cfg: Dict[str, Any]) -> None:
    if cfg["moe"].get("enabled", False):
        raise ValueError("BaseModel requires cfg.moe.enabled == False")
    _require_keys(cfg["model"], "model", ("encoder", "common", "fusion", "head"))
    _require_keys(cfg["model"]["head"], "model.head", ("type",))


def validate_bert_spc(cfg: Dict[str, Any]) -> None:
    if cfg["moe"].get("enabled", False):
        raise ValueError("BERTSPCModel requires cfg.moe.enabled == False")
    _require_keys(cfg["model"], "model", ("encoder", "common", "fusion", "head"))
    _require_keys(cfg["model"]["head"], "model.head", ("type",))
    _require_keys(cfg["model"], "model", ("bert_spc",))
    _require_keys(cfg["model"]["bert_spc"], "model.bert_spc", ("spc_type", "use_segment_ids"))


def validate_moeffn(cfg: Dict[str, Any]) -> None:
    if not cfg["moe"].get("enabled", False):
        raise ValueError("MoEFFN requires cfg.moe.enabled == True")
    num_experts = int(cfg["moe"].get("num_experts", 0))
    top_k = int(cfg["moe"].get("top_k", 0))
    if num_experts < 2:
        raise ValueError("MoEFFN requires cfg.moe.num_experts >= 2")
    if top_k < 1 or top_k > num_experts:
        raise ValueError("MoEFFN requires cfg.moe.top_k in [1, num_experts]")
    _require_keys(cfg["model"], "model", ("encoder", "common", "fusion", "head"))


def validate_moehead(cfg: Dict[str, Any]) -> None:
    if not cfg["moe"].get("enabled", False):
        raise ValueError("MoEHead requires cfg.moe.enabled == True")
    num_experts = int(cfg["moe"].get("num_experts", 0))
    top_k = int(cfg["moe"].get("top_k", 0))
    if num_experts < 2:
        raise ValueError("MoEHead requires cfg.moe.num_experts >= 2")
    if top_k < 1 or top_k > num_experts:
        raise ValueError("MoEHead requires cfg.moe.top_k in [1, num_experts]")
    _require_keys(cfg["model"], "model", ("encoder", "common", "fusion", "head"))
    _require_keys(cfg["model"]["head"], "model.head", ("type",))


def validate_moeskip(cfg: Dict[str, Any]) -> None:
    if not cfg["moe"].get("enabled", False):
        raise ValueError("MoESkip requires cfg.moe.enabled == True")
    num_experts = int(cfg["moe"].get("num_experts", 0))
    top_k = int(cfg["moe"].get("top_k", 0))
    if num_experts < 2:
        raise ValueError("MoESkip requires cfg.moe.num_experts >= 2")
    if top_k < 1 or top_k > num_experts:
        raise ValueError("MoESkip requires cfg.moe.top_k in [1, num_experts]")
    _require_keys(cfg["model"], "model", ("encoder", "common", "fusion", "head", "sk"))


def validate_hagmoe(cfg: Dict[str, Any]) -> None:
    if not cfg["moe"].get("enabled", False):
        raise ValueError("HAGMoE requires cfg.moe.enabled == True")
    _require_keys(cfg["model"], "model", ("encoder", "common", "fusion", "head", "hagmoe"))
    hag = cfg["model"].get("hagmoe", {})
    num_groups = int(hag.get("num_groups", 0))
    if num_groups < 2:
        raise ValueError("HAGMoE requires cfg.model.hagmoe.num_groups >= 2")
    top_k_groups = hag.get("router_topk_groups")
    if top_k_groups is None:
        top_k_groups = hag.get("top_k")
    if top_k_groups is None:
        top_k_groups = cfg.get("moe", {}).get("router", {}).get("top_k")
    if top_k_groups is not None:
        top_k_groups = int(top_k_groups)
        if top_k_groups < 0:
            raise ValueError("HAGMoE requires router_topk_groups >= 0")
        if top_k_groups > 0 and top_k_groups > num_groups:
            raise ValueError("HAGMoE requires router_topk_groups <= num_groups for group routing")
    norm = hag.get("top_k_normalization") or cfg.get("moe", {}).get("router", {}).get(
        "top_k_normalization"
    )
    if norm is not None:
        norm_val = str(norm).strip().lower()
        if norm_val not in {"renorm", "softmax"}:
            raise ValueError("HAGMoE top_k_normalization must be 'renorm' or 'softmax'")
    if hag.get("use_group_loss"):
        labels = cfg["model"].get("labels", {})
        if labels.get("id2label") is None and labels.get("label2id") is None:
            _warn("HAGMoE group loss enabled but labels are missing; will rely on runtime label mapping.")


def validate_config(cfg: Any) -> Any:
    """
    Validate config against the canonical schema.
    Returns cfg unchanged but raises if validation fails.
    """
    nested = _map_legacy_keys(cfg)

    _require_keys(nested, "root", ("model", "moe", "training", "data", "loss", "optim"))
    _require_keys(nested["model"], "model", ("name", "encoder", "common", "fusion"))
    _require_keys(nested["model"]["encoder"], "model.encoder", ("pretrained_name", "max_len", "dropout"))
    _require_keys(nested["model"]["common"], "model.common", ("num_labels",))
    _require_keys(nested["model"]["fusion"], "model.fusion", ("type", "dropout"))
    _require_keys(nested["moe"], "moe", ("enabled", "num_experts", "top_k"))
    _require_keys(nested["training"], "training", ("epochs", "batch_size", "eval_batch_size", "test_batch_size"))
    _require_keys(nested["data"], "data", ("train_path", "test_path", "max_len_sent", "max_len_term"))
    _require_keys(nested["loss"], "loss", ("type",))
    _require_keys(nested["optim"], "optim", ("lr", "lr_head", "warmup_ratio"))

    name = str(nested["model"].get("name") or "").strip()
    if name and name not in MODEL_NAMES:
        raise ValueError(f"cfg.model.name must be one of {MODEL_NAMES}, got '{name}'")

    if name == "BaseModel":
        validate_base_model(nested)
    elif name == "BERTSPCModel":
        validate_bert_spc(nested)
    elif name == "MoEFFN":
        validate_moeffn(nested)
    elif name == "MoEHead":
        validate_moehead(nested)
    elif name == "MoESkip":
        validate_moeskip(nested)
    elif name == "HAGMoE":
        validate_hagmoe(nested)

    # attach normalized nested config for downstream use (non-breaking)
    try:
        setattr(cfg, "_schema_cfg", nested)
    except Exception:
        pass

    return cfg
