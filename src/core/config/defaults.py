"""
Default configuration for a minimal BaseModel run.
These defaults are intentionally conservative and require the user to supply
dataset paths before training.
"""


def get_default_config() -> dict:
    return {
        "model": {
            "name": "BaseModel",
            "encoder": {
                "pretrained_name": "bert-base-uncased",
                "max_len": 128,
                "dropout": 0.1,
                "freeze": False,
            },
            "common": {"num_labels": 3, "hidden_size": 768},
            "fusion": {"type": "concat", "dropout": 0.1},
            "bert_spc": {"spc_type": "default", "use_segment_ids": True},
            "head": {"type": "linear", "mlp_layers": 2, "dropout": 0.1},
            "sk": {"mode": "add", "alpha_init": 0.5, "normalize": True},
            "hagmoe": {
                "group_strategy": "by_aspect",
                "num_groups": 2,
                "group_embed_dim": 64,
                "router_level": "aspect",
                "share_experts_across_groups": True,
                "expert_selection": "global_topk",
                "temperature": 1.0,
                "router_temperature": 1.0,
                "router_topk_groups": 0,
                "router_topk_apply_in_eval": True,
                "top_k": 0,
                "top_k_normalization": "renorm",
            },
        },
        "moe": {
            "enabled": False,
            "num_experts": 4,
            "top_k": 2,
            "capacity_factor": 1.0,
            "router": {
                "type": "softmax",
                "noise_std": 1.0,
                "temperature": 1.0,
                "top_k": 0,
                "top_k_normalization": "renorm",
            },
            "load_balance": {"enabled": True, "coef": 0.01, "type": "aux_loss"},
            "expert": {"ffn_dim": 3072, "dropout": 0.1, "activation": "gelu"},
        },
        "training": {
            "epochs": 1,
            "batch_size": 16,
            "eval_batch_size": 32,
            "test_batch_size": 32,
            "freeze_epochs": 0,
            "early_stop_patience": 0,
            "seed": 42,
            "device": "cuda",
        },
        "data": {
            "train_path": "",
            "test_path": "",
            "max_len_sent": 36,
            "max_len_term": 4,
            "val_path": None,
        },
        "loss": {"type": "ce", "class_weights": None, "focal_gamma": 2.0},
        "optim": {
            "lr": 2e-5,
            "lr_head": 1e-4,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "adamw_foreach": False,
            "adamw_fused": False,
        },
    }
