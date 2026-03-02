"""
Canonical configuration schema for the project.

This module documents required keys, optional keys, allowed enums, and expected types.
The schema is expressed as nested dicts with metadata to keep it lightweight and
human-readable. Validation logic lives in core/config/validate.py.
"""

MODEL_NAMES = (
    "BaseModel",
    "BERTSPCModel",
    "MoEFFN",
    "MoEHead",
    "MoESkip",
    "HAGMoE",
)

FUSION_TYPES = (
    "add",
    "concat",
    "gate_concat",
    "mul",
    "coattn",
    "crossattn",
    "late_interaction",
)

ROUTER_TYPES = ("softmax", "gated", "noisy_topk")
LOAD_BALANCE_TYPES = ("aux_loss", "cv_squared")
ACTIVATIONS = ("gelu", "relu", "silu", "tanh")
TOPK_NORMALIZATIONS = ("renorm", "softmax")

SCHEMA = {
    "model": {
        "required": ("name", "encoder", "common", "fusion"),
        "optional": ("bert_spc", "head", "sk", "hagmoe", "labels"),
        "types": {
            "name": str,
            "encoder": dict,
            "common": dict,
            "fusion": dict,
            "bert_spc": dict,
            "head": dict,
            "sk": dict,
            "hagmoe": dict,
            "labels": dict,
        },
        "enums": {"name": MODEL_NAMES},
        "encoder": {
            "required": ("pretrained_name", "max_len", "dropout", "freeze"),
            "types": {
                "pretrained_name": str,
                "max_len": int,
                "dropout": float,
                "freeze": bool,
            },
        },
        "common": {
            "required": ("num_labels", "hidden_size"),
            "types": {"num_labels": int, "hidden_size": int},
        },
        "fusion": {
            "required": ("type", "dropout"),
            "types": {"type": str, "dropout": float},
            "enums": {"type": FUSION_TYPES},
        },
        "bert_spc": {
            "required": ("spc_type", "use_segment_ids"),
            "types": {"spc_type": str, "use_segment_ids": bool},
        },
        "head": {
            "required": ("type",),
            "optional": ("mlp_layers", "dropout"),
            "types": {"type": str, "mlp_layers": int, "dropout": float},
        },
        "sk": {
            "required": ("mode",),
            "optional": ("alpha_init", "normalize", "beta_start", "beta_end", "beta_warmup_steps", "expert_hidden"),
            "types": {
                "mode": str,
                "alpha_init": float,
                "normalize": bool,
                "beta_start": float,
                "beta_end": float,
                "beta_warmup_steps": int,
                "expert_hidden": int,
            },
        },
        "hagmoe": {
            "required": (
                "num_groups",
            ),
            "optional": (
                "group_strategy",
                "group_embed_dim",
                "router_level",
                "share_experts_across_groups",
                "expert_selection",
                "temperature",
                "experts_per_group",
                "router_temperature",
                "group_temperature",
                "group_temperature_anneal",
                "top_k",
                "top_k_normalization",
                "router_topk_groups",
                "router_topk_apply_in_eval",
                "merge",
                "fusion_method",
                "use_group_loss",
                "use_balance_loss",
                "use_diversity_loss",
                "lambda_group",
                "lambda_balance",
                "lambda_diversity",
                "verbose_loss",
            ),
            "types": {
                "group_strategy": str,
                "num_groups": int,
                "group_embed_dim": int,
                "router_level": str,
                "share_experts_across_groups": bool,
                "expert_selection": str,
                "temperature": float,
                "experts_per_group": int,
                "router_temperature": float,
                "group_temperature": float,
                "group_temperature_anneal": str,
                "top_k": (int, type(None)),
                "top_k_normalization": str,
                "router_topk_groups": int,
                "router_topk_apply_in_eval": bool,
                "merge": str,
                "fusion_method": str,
                "use_group_loss": bool,
                "use_balance_loss": bool,
                "use_diversity_loss": bool,
                "lambda_group": float,
                "lambda_balance": float,
                "lambda_diversity": float,
                "verbose_loss": bool,
            },
            "enums": {
                "group_strategy": ("by_aspect", "by_pos", "by_span", "custom"),
                "router_level": ("token", "span", "aspect", "group"),
                "expert_selection": ("global_topk", "per_group_topk"),
                "top_k_normalization": TOPK_NORMALIZATIONS,
            },
        },
    },
    "moe": {
        "required": ("enabled", "num_experts", "top_k", "capacity_factor", "router", "load_balance", "expert"),
        "types": {
            "enabled": bool,
            "num_experts": int,
            "top_k": int,
            "capacity_factor": float,
            "router": dict,
            "load_balance": dict,
            "expert": dict,
        },
        "router": {
            "required": ("type", "noise_std", "temperature"),
            "optional": ("top_k", "top_k_normalization"),
            "types": {
                "type": str,
                "noise_std": float,
                "temperature": float,
                "top_k": (int, type(None)),
                "top_k_normalization": str,
            },
            "enums": {"type": ROUTER_TYPES},
        },
        "load_balance": {
            "required": ("enabled", "coef", "type"),
            "types": {"enabled": bool, "coef": float, "type": str},
            "enums": {"type": LOAD_BALANCE_TYPES},
        },
        "expert": {
            "required": ("ffn_dim", "dropout", "activation"),
            "types": {"ffn_dim": int, "dropout": float, "activation": str},
            "enums": {"activation": ACTIVATIONS},
        },
    },
    "training": {
        "required": ("epochs", "batch_size", "eval_batch_size", "test_batch_size"),
        "optional": ("freeze_epochs", "early_stop_patience", "seed", "device"),
        "types": {
            "epochs": int,
            "batch_size": int,
            "eval_batch_size": int,
            "test_batch_size": int,
            "freeze_epochs": int,
            "early_stop_patience": int,
            "seed": int,
            "device": str,
        },
    },
    "data": {
        "required": ("train_path", "test_path", "max_len_sent", "max_len_term"),
        "optional": ("val_path",),
        "types": {
            "train_path": str,
            "test_path": str,
            "max_len_sent": int,
            "max_len_term": int,
            "val_path": (str, type(None)),
        },
    },
    "loss": {
        "required": ("type",),
        "optional": ("class_weights", "focal_gamma"),
        "types": {"type": str, "class_weights": (list, type(None)), "focal_gamma": float},
        "enums": {"type": ("ce", "weighted_ce", "focal")},
    },
    "optim": {
        "required": ("lr", "lr_head", "warmup_ratio"),
        "optional": ("weight_decay", "adamw_foreach", "adamw_fused"),
        "types": {
            "lr": float,
            "lr_head": float,
            "warmup_ratio": float,
            "weight_decay": float,
            "adamw_foreach": bool,
            "adamw_fused": bool,
        },
    },
}
