from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional, Sequence, get_origin, get_args, Union


@dataclass
class EncoderConfig:
    pretrained_name: str = "bert-base-uncased"
    max_len: int = 128
    dropout: float = 0.1
    freeze: bool = False


@dataclass
class ModelCommonConfig:
    num_labels: int = 3
    hidden_size: int = 768


@dataclass
class FusionConfig:
    type: str = "concat"
    dropout: float = 0.1


@dataclass
class HeadConfig:
    type: str = "linear"
    mlp_layers: int = 2
    dropout: float = 0.1


@dataclass
class BertSpcConfig:
    spc_type: str = "default"
    use_segment_ids: bool = True


@dataclass
class SkConfig:
    mode: str = "add"
    alpha_init: float = 0.5
    normalize: bool = True
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_warmup_steps: int = 0
    expert_hidden: Optional[int] = None


@dataclass
class HagMoEConfig:
    num_groups: int = 3
    experts_per_group: int = 8
    router_temperature: float = 1.0
    group_temperature: float = 1.0
    group_temperature_anneal: str = ""
    top_k: Optional[int] = None
    top_k_normalization: str = "renorm"
    merge: str = "residual"
    fusion_method: str = "concat"
    use_group_loss: bool = True
    use_balance_loss: bool = True
    use_diversity_loss: bool = True
    lambda_group: float = 0.5
    lambda_balance: float = 0.01
    lambda_diversity: float = 0.2
    verbose_loss: bool = False


@dataclass
class LabelsConfig:
    id2label: Optional[dict] = None
    label2id: Optional[dict] = None


@dataclass
class ModelConfig:
    name: str = "BaseModel"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    common: ModelCommonConfig = field(default_factory=ModelCommonConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    bert_spc: BertSpcConfig = field(default_factory=BertSpcConfig)
    sk: SkConfig = field(default_factory=SkConfig)
    hagmoe: HagMoEConfig = field(default_factory=HagMoEConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)


@dataclass
class MoeRouterConfig:
    type: str = "softmax"
    noise_std: float = 0.001
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_k_normalization: str = "renorm"
    bias: bool = True
    entropy_weight: float = 0.0
    entropy_target: Optional[float] = None
    collapse_weight: float = 0.0
    collapse_tau: float = 0.02
    aux_warmup_steps: int = 0
    jitter_warmup_steps: int = 0
    jitter_end: float = 0.0


@dataclass
class MoeLoadBalanceConfig:
    enabled: bool = True
    coef: float = 0.01
    type: str = "aux_loss"


@dataclass
class MoeExpertConfig:
    ffn_dim: int = 3072
    dropout: float = 0.1
    activation: str = "gelu"


@dataclass
class MoeConfig:
    enabled: bool = False
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: Optional[float] = None
    route_mask_pad_tokens: bool = True
    router: MoeRouterConfig = field(default_factory=MoeRouterConfig)
    load_balance: MoeLoadBalanceConfig = field(default_factory=MoeLoadBalanceConfig)
    expert: MoeExpertConfig = field(default_factory=MoeExpertConfig)


@dataclass
class TrainingConfig:
    epochs: int = 15
    batch_size: int = 16
    eval_batch_size: int = 32
    test_batch_size: int = 32
    freeze_epochs: int = 3
    early_stop_patience: int = 8
    seed: int = 42
    num_seeds: int = 5
    seed_list: Optional[list[int]] = None
    shuffle: bool = True
    num_workers: int = 4
    max_grad_norm: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "fp16"
    step_print_moe: float = 100.0
    do_ensemble_logits: bool = True


@dataclass
class DataConfig:
    train_path: str = ""
    test_path: str = ""
    max_len_sent: int = 36
    max_len_term: int = 4
    val_path: Optional[str] = None


@dataclass
class LossConfig:
    type: str = "ce"
    class_weights: Optional[Sequence[float]] = None
    focal_gamma: float = 2.0


@dataclass
class OptimConfig:
    lr: float = 2e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    adamw_foreach: bool = False
    adamw_fused: bool = False


_LEGACY_DEFAULTS = {
    "model_name": "bert-base-uncased",
    "train_path": "",
    "test_path": "",
    "max_len_sent": 36,
    "max_len_term": 4,
    "num_labels": 3,
    "num_workers": 4,
    "id2label": None,
    "label2id": None,
    "epochs": 15,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "test_batch_size": 32,
    "lr": 2e-5,
    "lr_head": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "dropout": 0.1,
    "max_grad_norm": 1.0,
    "freeze_epochs": 3,
    "rolling_k": 3,
    "early_stop_patience": 8,
    "mode": "BaseModel",
    "fusion_method": "concat",
    "benchmark_fusions": False,
    "benchmark_methods": "sent,term,concat,add,mul,cross,gated_concat,bilinear,coattn,late_interaction",
    "train_full_only": False,
    "run_mode": "single",
    "seed": 42,
    "k_folds": 5,
    "num_seeds": 5,
    "seed_list": None,
    "shuffle": True,
    "output_dir": "results",
    "output_name": "results.json",
    "verbose_report": False,
    "val_path": None,
    "loss_type": "ce",
    "class_weights": None,
    "focal_gamma": 2.0,
    "use_amp": True,
    "amp_dtype": "fp16",
    "adamw_foreach": False,
    "adamw_fused": False,
    "debug_aspect_span": False,
    "freeze_moe": False,
    "aux_loss_weight": 0.01,
    "aux_warmup_steps": 0,
    "step_print_moe": 100,
    "do_ensemble_logits": True,
    "head_type": "linear",
    "num_experts": 8,
    "moe_top_k": 2,
    "router_bias": True,
    "router_jitter": 0.001,
    "jitter_warmup_steps": 0,
    "jitter_end": 0.0,
    "router_entropy_weight": 0.0,
    "route_mask_pad_tokens": True,
    "router_temperature": 1.0,
    "capacity_factor": None,
    "expert_hidden": None,
    "beta_start": 0.0,
    "beta_end": 1.0,
    "beta_warmup_steps": 0,
    "hag_num_groups": 3,
    "hag_experts_per_group": 8,
    "hag_router_temperature": 1.0,
    "hag_group_temperature": 1.0,
    "hag_group_temperature_anneal": "",
    "hag_merge": "residual",
    "hag_fusion_method": "concat",
    "hag_use_group_loss": True,
    "hag_use_balance_loss": True,
    "hag_use_diversity_loss": True,
    "hag_lambda_group": 0.5,
    "hag_lambda_balance": 0.01,
    "hag_lambda_diversity": 0.2,
    "hag_verbose_loss": False,
}


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    moe: MoeConfig = field(default_factory=MoeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    output_dir: str = "results"
    output_name: str = "results.json"
    fusion_method: str = "concat"
    benchmark_methods: str = _LEGACY_DEFAULTS["benchmark_methods"]
    run_mode: str = "single"
    debug_aspect_span: bool = False
    freeze_moe: bool = False

    legacy: dict = field(default_factory=dict, repr=False)

    # ----------------- legacy accessors -----------------
    _ALIASES = {
        "mode": ("model", "name"),
        "model_name": ("model", "encoder", "pretrained_name"),
        "num_labels": ("model", "common", "num_labels"),
        "dropout": ("model", "encoder", "dropout"),
        "head_type": ("model", "head", "type"),
        "train_path": ("data", "train_path"),
        "test_path": ("data", "test_path"),
        "max_len_sent": ("data", "max_len_sent"),
        "max_len_term": ("data", "max_len_term"),
        "val_path": ("data", "val_path"),
        "epochs": ("training", "epochs"),
        "train_batch_size": ("training", "batch_size"),
        "eval_batch_size": ("training", "eval_batch_size"),
        "test_batch_size": ("training", "test_batch_size"),
        "freeze_epochs": ("training", "freeze_epochs"),
        "early_stop_patience": ("training", "early_stop_patience"),
        "seed": ("training", "seed"),
        "num_seeds": ("training", "num_seeds"),
        "seed_list": ("training", "seed_list"),
        "shuffle": ("training", "shuffle"),
        "num_workers": ("training", "num_workers"),
        "max_grad_norm": ("training", "max_grad_norm"),
        "use_amp": ("training", "use_amp"),
        "amp_dtype": ("training", "amp_dtype"),
        "step_print_moe": ("training", "step_print_moe"),
        "do_ensemble_logits": ("training", "do_ensemble_logits"),
        "loss_type": ("loss", "type"),
        "class_weights": ("loss", "class_weights"),
        "focal_gamma": ("loss", "focal_gamma"),
        "lr": ("optim", "lr"),
        "lr_head": ("optim", "lr_head"),
        "weight_decay": ("optim", "weight_decay"),
        "warmup_ratio": ("optim", "warmup_ratio"),
        "adamw_foreach": ("optim", "adamw_foreach"),
        "adamw_fused": ("optim", "adamw_fused"),
        "route_mask_pad_tokens": ("moe", "route_mask_pad_tokens"),
        "num_experts": ("moe", "num_experts"),
        "moe_top_k": ("moe", "top_k"),
        "capacity_factor": ("moe", "capacity_factor"),
        "router_bias": ("moe", "router", "bias"),
        "router_jitter": ("moe", "router", "noise_std"),
        "router_temperature": ("moe", "router", "temperature"),
        "router_entropy_weight": ("moe", "router", "entropy_weight"),
        "router_entropy_target": ("moe", "router", "entropy_target"),
        "router_collapse_weight": ("moe", "router", "collapse_weight"),
        "router_collapse_tau": ("moe", "router", "collapse_tau"),
        "aux_warmup_steps": ("moe", "router", "aux_warmup_steps"),
        "jitter_warmup_steps": ("moe", "router", "jitter_warmup_steps"),
        "jitter_end": ("moe", "router", "jitter_end"),
        "aux_loss_weight": ("moe", "load_balance", "coef"),
        "beta_start": ("model", "sk", "beta_start"),
        "beta_end": ("model", "sk", "beta_end"),
        "beta_warmup_steps": ("model", "sk", "beta_warmup_steps"),
        "expert_hidden": ("model", "sk", "expert_hidden"),
        "hag_num_groups": ("model", "hagmoe", "num_groups"),
        "hag_experts_per_group": ("model", "hagmoe", "experts_per_group"),
        "hag_router_temperature": ("model", "hagmoe", "router_temperature"),
        "hag_group_temperature": ("model", "hagmoe", "group_temperature"),
        "hag_group_temperature_anneal": ("model", "hagmoe", "group_temperature_anneal"),
        "hag_merge": ("model", "hagmoe", "merge"),
        "hag_fusion_method": ("model", "hagmoe", "fusion_method"),
        "hag_use_group_loss": ("model", "hagmoe", "use_group_loss"),
        "hag_use_balance_loss": ("model", "hagmoe", "use_balance_loss"),
        "hag_use_diversity_loss": ("model", "hagmoe", "use_diversity_loss"),
        "hag_lambda_group": ("model", "hagmoe", "lambda_group"),
        "hag_lambda_balance": ("model", "hagmoe", "lambda_balance"),
        "hag_lambda_diversity": ("model", "hagmoe", "lambda_diversity"),
        "hag_verbose_loss": ("model", "hagmoe", "verbose_loss"),
        "id2label": ("model", "labels", "id2label"),
        "label2id": ("model", "labels", "label2id"),
    }

    def __getattr__(self, name: str):
        if name in self._ALIASES:
            return _get_nested(self, self._ALIASES[name])
        if "legacy" in self.__dict__ and name in self.__dict__["legacy"]:
            return self.__dict__["legacy"][name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name in {
            "model",
            "moe",
            "training",
            "data",
            "loss",
            "optim",
            "output_dir",
            "output_name",
            "fusion_method",
            "benchmark_methods",
            "run_mode",
            "debug_aspect_span",
            "freeze_moe",
            "legacy",
        }:
            super().__setattr__(name, value)
            return
        if name in self._ALIASES:
            _set_nested(self, self._ALIASES[name], value)
            return
        if "legacy" in self.__dict__:
            self.__dict__["legacy"][name] = value
            return
        super().__setattr__(name, value)

    # ----------------- factory -----------------
    @staticmethod
    def _is_bool_type(t) -> bool:
        if t is bool:
            return True
        origin = get_origin(t)
        if origin is Union:
            args = [a for a in get_args(t) if a is not type(None)]
            return len(args) == 1 and args[0] is bool
        return False

    @staticmethod
    def _unwrap_optional(t):
        origin = get_origin(t)
        if origin is Union:
            args = [a for a in get_args(t) if a is not type(None)]
            return args[0] if len(args) == 1 else t
        return t

    @classmethod
    def from_cli(cls, argv=None):
        parser = argparse.ArgumentParser("ATSC Trainer")

        defaults_cfg = cls()
        for name, legacy_default in _LEGACY_DEFAULTS.items():
            default = getattr(defaults_cfg, name, legacy_default)
            hinted_type = type(default) if default is not None else str
            hinted_type = cls._unwrap_optional(hinted_type)

            if cls._is_bool_type(hinted_type):
                group = parser.add_mutually_exclusive_group(required=False)
                group.add_argument(f"--{name}", dest=name, action="store_true")
                group.add_argument(f"--no_{name}", dest=name, action="store_false")
                parser.set_defaults(**{name: bool(default)})
                continue

            if name == "seed_list":
                parser.add_argument(f"--{name}", nargs="*", type=int, default=default)
                continue

            if default is None:
                parser.add_argument(f"--{name}", default=None)
            else:
                parser.add_argument(f"--{name}", type=type(default), default=default)

        ns = parser.parse_args(argv)
        return cls.from_legacy(vars(ns))

    @classmethod
    def from_legacy(cls, legacy: dict) -> "Config":
        cfg = cls()
        cfg.legacy = dict(legacy)
        for key, value in legacy.items():
            setattr(cfg, key, value)
        return cfg

    # ----------------- helpers -----------------
    @property
    def is_benchmark(self) -> bool:
        return bool(self.benchmark_methods)

    def finalize(self) -> "Config":
        # class_weights: "1.0,2.5,1.0" -> [1.0,2.5,1.0]
        if isinstance(self.class_weights, str):
            s = self.class_weights.strip()
            self.class_weights = None if not s else [float(x.strip()) for x in s.split(",")]

        self.benchmark_methods = (self.benchmark_methods or "").strip()
        if not self.output_name:
            self.output_name = "results.json"

        if self.num_seeds < 1:
            self.num_seeds = 1

        if self.seed_list is None or len(self.seed_list) == 0:
            self.seed_list = [self.seed + i for i in range(self.num_seeds)]
        else:
            self.seed_list = [int(x) for x in self.seed_list]

        if not self.seed_list:
            self.seed_list = [self.seed]
        return self

    def validate(self) -> "Config":
        if self.loss_type == "focal" and (self.focal_gamma is None):
            raise ValueError("loss_type=focal requires focal_gamma")
        if getattr(self, "k_folds", 0) < 0:
            raise ValueError("k_folds must be >= 0")
        if not self.train_path:
            raise ValueError("train_path is required")
        if not self.test_path:
            raise ValueError("test_path is required")

        from .validate import validate_config

        validate_config(self)
        return self


def _get_nested(cfg: Config, path: tuple[str, ...]):
    cur = cfg
    for key in path:
        cur = getattr(cur, key)
    return cur


def _set_nested(cfg: Config, path: tuple[str, ...], value) -> None:
    cur = cfg
    for key in path[:-1]:
        cur = getattr(cur, key)
    setattr(cur, path[-1], value)


from .defaults import get_default_config  # noqa: E402
from .validate import validate_config  # noqa: E402

__all__ = [
    "Config",
    "ModelConfig",
    "MoeConfig",
    "TrainingConfig",
    "DataConfig",
    "LossConfig",
    "OptimConfig",
    "get_default_config",
    "validate_config",
]
