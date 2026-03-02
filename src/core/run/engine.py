from collections import deque
import ast
import json
import os
from datetime import datetime
import re
import string
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from torch.amp import autocast, GradScaler

from src.core.utils.const import DEVICE
from src.core.utils.general import cleanup_cuda, safe_float
from src.core.utils.optim import build_optimizer_and_scheduler
from src.core.utils.plotting import print_confusion_matrix
from src.core.utils.moe_metrics import MoEMetricsAccumulator
from src.core.utils.calibration import compute_calibration
from src.core.utils.artifacts import save_artifacts


class RoutingEpochAggregator:
    def __init__(self) -> None:
        self.sum_ent = 0.0
        self.sum_ent2 = 0.0
        self.count_samples = 0
        self.sum_group_usage: Optional[torch.Tensor] = None
        self.sum_top1_max = 0.0
        self.top1_counts: Optional[torch.Tensor] = None
        self.nan_batches = 0
        self.num_groups: Optional[int] = None
        self.enabled = False
        self.batch_count_logged = 0
        self.min_group_mean_min = float("inf")
        self.min_group_mean_sum = 0.0
        self.min_group_mean_max = float("-inf")
        self.max_group_mean_min = float("inf")
        self.max_group_mean_sum = 0.0
        self.max_group_mean_max = float("-inf")
        self.ent_of_mean_min = float("inf")
        self.ent_of_mean_sum = 0.0
        self.ent_of_mean_max = float("-inf")
        self.low_min_group_mean_count = 0
        self.collapse_tau = None

    def update(self, outputs: Optional[Dict[str, Any]], model: nn.Module) -> None:
        p_group = None
        if isinstance(outputs, dict) and "group_probs" in outputs:
            p_group = outputs.get("group_probs")
        if p_group is None and hasattr(model, "_last_group_probs"):
            p_group = getattr(model, "_last_group_probs", None)
        if p_group is None:
            return
        if not torch.is_tensor(p_group) or p_group.numel() == 0 or p_group.dim() != 2:
            return

        self.enabled = True

        if not torch.isfinite(p_group).all():
            self.nan_batches += 1
            return

        p_group = p_group.detach()
        if p_group.device.type != "cpu":
            p_group = p_group.to("cpu")

        bsz, num_groups = p_group.shape
        if self.num_groups is None:
            self.num_groups = int(num_groups)
            self.sum_group_usage = torch.zeros(num_groups, dtype=torch.float64)
            self.top1_counts = torch.zeros(num_groups, dtype=torch.int64)
        elif int(num_groups) != int(self.num_groups):
            return

        p = p_group.clamp_min(1e-12)
        ent = -(p * p.log()).sum(dim=-1)
        top1_max = p_group.max(dim=-1).values
        top1 = p_group.argmax(dim=-1)

        group_mean_batch = p_group.mean(dim=0)
        min_gm = float(group_mean_batch.min().item())
        max_gm = float(group_mean_batch.max().item())
        mean_p = group_mean_batch.clamp_min(1e-12)
        ent_of_mean = float(-(mean_p * mean_p.log()).sum().item())
        if self.collapse_tau is None:
            try:
                self.collapse_tau = float(getattr(model, "router_collapse_tau", 0.01))
            except Exception:
                self.collapse_tau = 0.01

        self.batch_count_logged += 1
        self.min_group_mean_sum += min_gm
        self.min_group_mean_min = min(self.min_group_mean_min, min_gm)
        self.min_group_mean_max = max(self.min_group_mean_max, min_gm)
        self.max_group_mean_sum += max_gm
        self.max_group_mean_min = min(self.max_group_mean_min, max_gm)
        self.max_group_mean_max = max(self.max_group_mean_max, max_gm)
        self.ent_of_mean_sum += ent_of_mean
        self.ent_of_mean_min = min(self.ent_of_mean_min, ent_of_mean)
        self.ent_of_mean_max = max(self.ent_of_mean_max, ent_of_mean)
        tau = self.collapse_tau if self.collapse_tau is not None else 0.01
        if min_gm < tau:
            self.low_min_group_mean_count += 1

        self.sum_ent += float(ent.sum().item())
        self.sum_ent2 += float((ent * ent).sum().item())
        self.count_samples += int(bsz)
        self.sum_top1_max += float(top1_max.sum().item())

        if self.sum_group_usage is not None:
            self.sum_group_usage += p_group.sum(dim=0).to(dtype=torch.float64)
        if self.top1_counts is not None:
            self.top1_counts += torch.bincount(
                top1.to(torch.int64), minlength=int(self.num_groups)
            ).to(dtype=torch.int64)

    def finalize(self) -> Optional[Dict[str, Any]]:
        if not self.enabled or self.count_samples <= 0 or self.sum_group_usage is None:
            return None
        mean_ent = self.sum_ent / self.count_samples
        var_ent = self.sum_ent2 / self.count_samples - mean_ent**2
        if var_ent < 0.0:
            var_ent = 0.0
        std_ent = float(var_ent ** 0.5)
        group_usage = (self.sum_group_usage / self.count_samples).tolist()
        top1_dominance_mean = self.sum_top1_max / self.count_samples
        if self.top1_counts is None:
            top1_hist = None
        else:
            top1_hist = (self.top1_counts.to(dtype=torch.float64) / self.count_samples).tolist()
        top1_dominance = None
        if top1_hist:
            top1_dominance = float(max(top1_hist))

        min_group_mean_stats = None
        max_group_mean_stats = None
        ent_of_mean_stats = None
        low_min_group_mean_ratio = None
        if self.batch_count_logged > 0:
            min_group_mean_stats = {
                "min": float(self.min_group_mean_min),
                "mean": float(self.min_group_mean_sum / self.batch_count_logged),
                "max": float(self.min_group_mean_max),
            }
            max_group_mean_stats = {
                "min": float(self.max_group_mean_min),
                "mean": float(self.max_group_mean_sum / self.batch_count_logged),
                "max": float(self.max_group_mean_max),
            }
            ent_of_mean_stats = {
                "min": float(self.ent_of_mean_min),
                "mean": float(self.ent_of_mean_sum / self.batch_count_logged),
                "max": float(self.ent_of_mean_max),
            }
            low_min_group_mean_ratio = float(
                self.low_min_group_mean_count / self.batch_count_logged
            )

        return {
            "mean_ent": float(mean_ent),
            "std_ent": float(std_ent),
            "group_usage": group_usage,
            "top1_dominance": top1_dominance,
            "top1_dominance_mean": float(top1_dominance_mean),
            "top1_hist": top1_hist,
            "nan_batches": int(self.nan_batches),
            "min_group_mean_batch_stats": min_group_mean_stats,
            "max_group_mean_batch_stats": max_group_mean_stats,
            "entropy_of_mean_batch_stats": ent_of_mean_stats,
            "low_min_group_mean_ratio": low_min_group_mean_ratio,
            "batch_count_logged": int(self.batch_count_logged),
            "collapse_tau": self.collapse_tau,
        }


def _extract_seed_fold(tag: Optional[str], cfg) -> tuple[str, str]:
    seed = None
    fold = None
    if tag:
        m_seed = re.search(r"seed\s*=\s*(\d+)", tag)
        if m_seed:
            seed = m_seed.group(1)
        m_fold = re.search(r"fold\s*=\s*(\d+)", tag)
        if m_fold:
            fold = m_fold.group(1)

    if seed is None:
        seed = str(getattr(cfg, "seed", "unknown"))
    if fold is None:
        fold = "full"
    return seed, fold


def _routing_log_path(cfg, tag: Optional[str], method: Optional[str]) -> str:
    output_dir = getattr(cfg, "output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    seed, fold = _extract_seed_fold(tag, cfg)
    mode = str(getattr(cfg, "mode", "unknown"))
    loss_type = str(getattr(cfg, "loss_type", "unknown"))
    method_str = str(method or "unknown")
    base = os.path.join(output_dir, mode, method_str, loss_type, f"seed_{seed}", f"fold_{fold}")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "routing_logs.jsonl")


def _append_routing_log(path: str, entry: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _normalize_aspect_text(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("-", " ")
    s = s.strip(string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _move_batch_to_device(batch_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch_dict.items():
        if torch.is_tensor(v):
            out[k] = v.to(DEVICE)
        else:
            out[k] = v
    return out


def _forward_step(
    cfg, model: nn.Module, batch: Dict[str, Any], fusion_method: str
) -> Dict[str, Any]:
    if cfg is not None and str(getattr(cfg, "mode", "")).strip() == "HAGMoE":
        return model(
            input_ids_sent=batch["input_ids_sent"],
            attention_mask_sent=batch["attention_mask_sent"],
            input_ids_term=batch["input_ids_term"],
            attention_mask_term=batch["attention_mask_term"],
            aspect_start=batch.get("aspect_start"),
            aspect_end=batch.get("aspect_end"),
            aspect_mask_sent=batch.get("aspect_mask_sent"),
            labels=batch["label"],
            fusion_method=fusion_method,
        )
    return model(
        input_ids_sent=batch["input_ids_sent"],
        attention_mask_sent=batch["attention_mask_sent"],
        input_ids_term=batch["input_ids_term"],
        attention_mask_term=batch["attention_mask_term"],
        labels=batch["label"],
        fusion_method=fusion_method,
    )


def _normalize_model_output(outputs: Any) -> tuple[torch.Tensor, Dict[str, Any]]:
    if isinstance(outputs, dict):
        if "logits" not in outputs:
            raise ValueError("Model output dict missing 'logits'")
        logits = outputs["logits"]
        extras = {k: v for k, v in outputs.items() if k != "logits"}
        return logits, extras
    if torch.is_tensor(outputs):
        return outputs, {}
    if isinstance(outputs, (tuple, list)) and outputs:
        logits = outputs[0]
        extras = outputs[1] if len(outputs) > 1 else {}
        if extras is None:
            extras = {}
        if not isinstance(extras, dict):
            extras = {"extras": extras}
        return logits, extras
    raise ValueError("Unsupported model output format")


def _backward_step(
    *,
    loss_total: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    use_amp: bool,
    max_grad_norm: Optional[float],
) -> None:
    if use_amp:
        if scaler is None:
            raise RuntimeError("use_amp=True but scaler is None")
        scaler.scale(loss_total).backward()

        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))

        scaler.step(optimizer)
        scaler.update()
    else:
        loss_total.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()

    if scheduler is not None:
        scheduler.step()


def _accumulate_basic_metrics(
    *,
    counters: Dict[str, Any],
    loss_total: torch.Tensor,
    loss_main,
    loss_lambda,
    loss_aux,
    moe: bool,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    counters["total_loss_sum"] += safe_float(loss_total)
    if moe:
        counters["main_loss_sum"] += safe_float(loss_main)
        counters["lambda_loss_sum"] += safe_float(loss_lambda)
        counters["aux_loss_sum"] += safe_float(loss_aux)
    counters["n_steps"] += 1

    preds = torch.argmax(logits, dim=-1)
    counters["all_preds"].extend(preds.detach().cpu().tolist())
    counters["all_labels"].extend(labels.detach().cpu().tolist())


def _init_hag_log_buffers() -> tuple[Dict[str, float], Dict[str, int]]:
    sums = {
        "loss": 0.0,
        "loss_main": 0.0,
        "aux_loss": 0.0,
        "loss_group": 0.0,
        "loss_balance": 0.0,
        "loss_diversity": 0.0,
        "loss_entropy_raw": 0.0,
        "loss_entropy_used": 0.0,
        "loss_collapse_raw": 0.0,
        "loss_collapse_used": 0.0,
    }
    counts = {k: 0 for k in sums}
    return sums, counts


def _update_hag_log_buffers(
    outputs: Dict[str, Any],
    hag_log_sums: Dict[str, float],
    hag_log_counts: Dict[str, int],
) -> None:
    for key in hag_log_sums:
        if key in outputs and outputs.get(key) is not None:
            hag_log_sums[key] += safe_float(outputs.get(key))
            hag_log_counts[key] += 1


def _print_hag_epoch_loss_summary(
    *,
    model: nn.Module,
    hag_log_sums: Dict[str, float],
    hag_log_counts: Dict[str, int],
) -> None:
    parts = ["epoch_summary"]
    for key, total in hag_log_sums.items():
        cnt = max(1, hag_log_counts[key])
        if hag_log_counts[key] > 0:
            parts.append(f"{key}={total / cnt:.6f}")
    print("[HAGMoE] " + " ".join(parts))
    entropy_weight = float(getattr(model, "router_entropy_weight", 0.0) or 0.0)
    ent_cnt = hag_log_counts.get("loss_entropy_used", 0)
    if entropy_weight > 0.0 and ent_cnt > 0:
        ent_mean = hag_log_sums.get("loss_entropy_used", 0.0) / max(1, ent_cnt)
        if abs(ent_mean) < 1e-8:
            print("[HAGMoE] Warning: router_entropy_weight > 0 but loss_entropy_used ~ 0.")


def _init_aspect_span_counters() -> Dict[str, Any]:
    return {
        "match_total": 0,
        "match_matched": 0,
        "match_mask_sum": 0.0,
        "match_zero": 0,
        "token_mismatch_count": 0,
        "truncated_count": 0,
        "not_found_raw_count": 0,
        "unknown_count": 0,
    }


def _aspect_span_update(
    batch: Dict[str, Any],
    cfg,
    counters: Dict[str, Any],
    *,
    verbose: bool,
    use_cfg_max_len: bool,
    epoch_idx: Optional[int] = None,
    split: Optional[str] = None,
    batch_idx: Optional[int] = None,
) -> None:
    if "aspect_mask_sent" not in batch:
        return

    mask_sum = batch["aspect_mask_sent"].detach().sum(dim=1)
    matched = mask_sum > 0
    counters["match_total"] += int(mask_sum.numel())
    counters["match_matched"] += int(matched.sum().item())
    counters["match_mask_sum"] += float(mask_sum[matched].sum().item())
    counters["match_zero"] += int((~matched).sum().item())

    if (
        "sentence_raw" in batch
        and "aspect_raw" in batch
        and "valid_len" in batch
        and "sep_idx" in batch
    ):
        sentence_raw = batch["sentence_raw"]
        aspect_raw = batch["aspect_raw"]
        valid_len = batch["valid_len"].detach().cpu().tolist()
        sep_idx = batch["sep_idx"].detach().cpu().tolist()
        max_len_sent = int(
            batch["max_len_sent"][0].item()
            if "max_len_sent" in batch and torch.is_tensor(batch["max_len_sent"])
            else (int(getattr(cfg, "max_len_sent", 0) or 0) if use_cfg_max_len else 0)
        )
        for i in range(len(mask_sum)):
            if matched[i]:
                continue
            try:
                sent_norm = _normalize_aspect_text(sentence_raw[i])
                asp_norm = _normalize_aspect_text(aspect_raw[i])
            except Exception:
                counters["unknown_count"] += 1
                continue
            raw_found = asp_norm != "" and asp_norm in sent_norm
            truncated = (valid_len[i] >= max_len_sent) or (sep_idx[i] >= max_len_sent - 1)
            if raw_found and truncated:
                counters["truncated_count"] += 1
            elif raw_found:
                counters["token_mismatch_count"] += 1
            else:
                counters["not_found_raw_count"] += 1
    else:
        counters["unknown_count"] += int((~matched).sum().item())

    if (
        verbose
        and epoch_idx == 0
        and split in {"val", "test"}
        and batch_idx == 0
    ):
        sentence_raw = batch.get("sentence_raw", [])
        aspect_raw = batch.get("aspect_raw", [])
        valid_len = (
            batch["valid_len"].detach().cpu().tolist()
            if "valid_len" in batch
            else [0] * len(mask_sum)
        )
        sep_idx = (
            batch["sep_idx"].detach().cpu().tolist()
            if "sep_idx" in batch
            else [-1] * len(mask_sum)
        )
        max_len_sent = int(
            batch["max_len_sent"][0].item()
            if "max_len_sent" in batch and torch.is_tensor(batch["max_len_sent"])
            else 0
        )

        for i in range(min(10, len(mask_sum))):
            if mask_sum[i].item() > 0:
                continue
            try:
                sent_norm = _normalize_aspect_text(sentence_raw[i])
                asp_norm = _normalize_aspect_text(aspect_raw[i])
            except Exception:
                sent_norm = ""
                asp_norm = ""
            raw_idx = sent_norm.find(asp_norm) if asp_norm else -1
            raw_found = raw_idx >= 0
            truncated = (valid_len[i] >= max_len_sent) or (sep_idx[i] >= max_len_sent - 1)
            if raw_found and truncated:
                reason = "TRUNCATED"
            elif raw_found:
                reason = "TOKEN_MISMATCH"
            else:
                reason = "NOT_FOUND_RAW"

            block = [
                f"[AspectSpanDebug] epoch={epoch_idx} split={split} batch={batch_idx} sample={i}",
                f"  max_len_sent={max_len_sent} valid_len={valid_len[i]} sep_idx={sep_idx[i]}",
                f"  sentence_raw: {sentence_raw[i] if i < len(sentence_raw) else ''}",
                f"  aspect_raw: {aspect_raw[i] if i < len(aspect_raw) else ''}",
                f"  sentence_norm: {sent_norm}",
                f"  aspect_norm: {asp_norm}",
                f"  aspect_mask_sum: {int(mask_sum[i].item())}",
                f"  raw_found_substring: {raw_found} idx={raw_idx}",
                f"  truncated: {truncated}",
                f"  fail_reason: {reason}",
            ]
            print("\n".join(block))


def _print_routing_epoch_summary(routing_metrics: Dict[str, Any], split: str) -> None:
    usage = routing_metrics.get("group_usage", [])
    top1_hist = routing_metrics.get("top1_hist", [])
    ent_stats = routing_metrics.get("entropy_of_mean_batch_stats")
    ent_str = "None"
    if ent_stats:
        ent_str = (
            f"[{ent_stats['min']:.4f}/"
            f"{ent_stats['mean']:.4f}/"
            f"{ent_stats['max']:.4f}]"
        )
    min_gm_stats = routing_metrics.get("min_group_mean_batch_stats")
    min_gm_str = "None"
    if min_gm_stats:
        min_gm_str = (
            f"[{min_gm_stats['min']:.4f}/"
            f"{min_gm_stats['mean']:.4f}/"
            f"{min_gm_stats['max']:.4f}]"
        )
    max_gm_stats = routing_metrics.get("max_group_mean_batch_stats")
    max_gm_str = "None"
    if max_gm_stats:
        max_gm_str = (
            f"[{max_gm_stats['min']:.4f}/"
            f"{max_gm_stats['mean']:.4f}/"
            f"{max_gm_stats['max']:.4f}]"
        )
    top1_dom = routing_metrics.get("top1_dominance")
    top1_dom_str = f"{top1_dom:.6f}" if top1_dom is not None else "None"
    print(
        f"Routing({split}): "
        f"mean_ent={routing_metrics['mean_ent']:.6f} "
        f"std_ent={routing_metrics['std_ent']:.6f} "
        f"usage={usage} "
        f"top1_dom={top1_dom_str} "
        f"top1_hist={top1_hist} "
        f"minGM[min/mean/max]={min_gm_str} "
        f"maxGM[min/mean/max]={max_gm_str} "
        f"lowMinGM={routing_metrics.get('low_min_group_mean_ratio')} "
        f"entMean[min/mean/max]={ent_str} "
        f"nan_batches={routing_metrics['nan_batches']}"
    )


def _write_routing_entry(
    *,
    cfg,
    tag: Optional[str],
    method: Optional[str],
    routing_metrics: Optional[Dict[str, Any]],
    split: str,
    epoch_idx: int,
    loss: Optional[float] = None,
    macro_f1: Optional[float] = None,
    neutral_f1: Optional[float] = None,
) -> None:
    if routing_metrics is None:
        return
    seed_str, fold_str = _extract_seed_fold(tag, cfg)
    try:
        seed_val: Any = int(seed_str)
    except Exception:
        seed_val = seed_str
    try:
        fold_val: Any = int(fold_str)
    except Exception:
        fold_val = fold_str

    entry = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed_val,
        "fold": fold_val,
        "epoch": int(epoch_idx) + 1,
        "split": str(split),
        "mean_ent": routing_metrics["mean_ent"],
        "std_ent": routing_metrics["std_ent"],
        "group_usage": routing_metrics["group_usage"],
        "top1_dominance": routing_metrics["top1_dominance"],
        "top1_dominance_mean": routing_metrics.get("top1_dominance_mean"),
        "top1_hist": routing_metrics["top1_hist"],
        "nan_batches": routing_metrics.get("nan_batches", 0),
        "min_group_mean_batch_stats": routing_metrics.get("min_group_mean_batch_stats"),
        "max_group_mean_batch_stats": routing_metrics.get("max_group_mean_batch_stats"),
        "entropy_of_mean_batch_stats": routing_metrics.get("entropy_of_mean_batch_stats"),
        "low_min_group_mean_ratio": routing_metrics.get("low_min_group_mean_ratio"),
        "batch_count_logged": routing_metrics.get("batch_count_logged"),
        "collapse_tau": routing_metrics.get("collapse_tau"),
    }
    if macro_f1 is not None:
        entry["macro_f1"] = float(macro_f1)
    if neutral_f1 is not None:
        entry["neutral_f1"] = float(neutral_f1)
    if loss is not None:
        entry["loss"] = float(loss)

    _append_routing_log(_routing_log_path(cfg, tag, method), entry)


def _normalize_schedule(raw) -> Optional[list]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            return None
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    return None


def _resolve_schedule(raw, epoch_num: int) -> Optional[Any]:
    sched = _normalize_schedule(raw)
    if not sched:
        return None
    current = None
    for item in sched:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            start = int(item[0])
        except Exception:
            continue
        if start <= epoch_num:
            current = item[1]
    return current


def _apply_hagmoe_router_schedules(cfg, model: nn.Module, epoch_idx: int) -> None:
    epoch_num = int(epoch_idx) + 1
    ent_w = _resolve_schedule(getattr(cfg, "router_entropy_schedule", None), epoch_num)
    if ent_w is not None and hasattr(model, "router_entropy_weight"):
        model.router_entropy_weight = float(ent_w)
    ent_t = _resolve_schedule(getattr(cfg, "router_entropy_target_schedule", None), epoch_num)
    if ent_t is not None and hasattr(model, "router_entropy_target"):
        model.router_entropy_target = ent_t
    col_w = _resolve_schedule(getattr(cfg, "router_collapse_schedule", None), epoch_num)
    if col_w is not None and hasattr(model, "router_collapse_weight"):
        model.router_collapse_weight = float(col_w)
    col_tau = _resolve_schedule(getattr(cfg, "router_collapse_tau_schedule", None), epoch_num)
    if col_tau is not None and hasattr(model, "router_collapse_tau"):
        model.router_collapse_tau = float(col_tau)


def _set_encoder_requires_grad(
    cfg,
    model: nn.Module,
    *,
    trainable: bool,
    keep_moe_trainable: bool,
) -> None:
    if not hasattr(model, "encoder"):
        return

    for name, p in model.encoder.named_parameters():
        if keep_moe_trainable and ("moe_ffn" in name):
            p.requires_grad = True
        else:
            p.requires_grad = bool(trainable)


def _set_encoder_train_eval(model: nn.Module, *, frozen: bool) -> None:
    if not hasattr(model, "encoder"):
        return
    model.encoder.eval() if frozen else model.encoder.train()


def _encoder_grad_summary(model: nn.Module) -> Optional[Dict[str, int]]:
    if not hasattr(model, "encoder"):
        return None
    total = 0
    trainable = 0
    for _, p in model.encoder.named_parameters():
        total += 1
        if p.requires_grad:
            trainable += 1
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def maybe_freeze_encoder(cfg, model: nn.Module, *, epoch_idx_0based: int) -> bool:
    """
    Schedule A (stable warmup):
    - For the first `cfg.freeze_epochs` epochs: freeze the entire encoder (BERT backbone),
      so only modules outside `model.encoder` are trained.
    - After that: unfreeze the encoder and train everything.

    Exception:
    - If `cfg.mode == "MoEFFN"`, we keep the existing specialized logic
      (MoE FFN lives inside the encoder and is handled there).
    """
    fe = cfg.freeze_epochs
    if fe <= 0:
        _set_encoder_requires_grad(cfg, model, trainable=True, keep_moe_trainable=False)
        _set_encoder_train_eval(model, frozen=False)
        return False

    in_freeze = epoch_idx_0based < fe
    mode = cfg.mode
    
    if in_freeze:
        # Keep the user's existing MoEFFN logic untouched.
        if mode == "MoEFFN":
            print("MoEFFN mode: freezing base encoder, keeping MoE FFN trainable")
            keep_moe = cfg.freeze_moe
            _set_encoder_requires_grad(cfg, model, trainable=False, keep_moe_trainable=keep_moe)
            _set_encoder_train_eval(model, frozen=True)
            return True
        
        if mode == "MoESkipModel":
            enc = getattr(model, "encoder", None)
            if enc is None:
                return False

            base = getattr(enc, "base_encoder", None)
            if base is None:
                print(f"{mode} mode: encoder has no base_encoder, freezing entire encoder")
                _set_encoder_requires_grad(cfg, model, trainable=False, keep_moe_trainable=True)
                _set_encoder_train_eval(model, frozen=True)
                return True

            print(f"{mode} mode: freezing base_encoder only, keeping MoE-skip + fusion modules trainable")

            # 1) Freeze backbone params
            for p in base.parameters():
                p.requires_grad = False
            base.eval()

            # 2) Keep MoE modules trainable
            for name, p in enc.named_parameters():
                if name.startswith("base_encoder."):
                    continue
                p.requires_grad = True
            
            # 3) Keep fusion modules trainable (nằm ngoài encoder)
            # Các modules này đã tự động trainable vì không thuộc base_encoder
            # Nhưng cần đảm bảo chúng ở train mode
            for name, module in model.named_modules():
                if name.startswith("encoder.base_encoder"):
                    continue
                if hasattr(module, 'train') and callable(module.train):
                    if name and not name.startswith("encoder.base_encoder"):
                        module.train()
            
            return True
              
        # General case: freeze the entire encoder (head-only warmup).
        print(f"{mode} mode: freezing entire encoder")
        _set_encoder_requires_grad(cfg, model, trainable=False, keep_moe_trainable=True)
        _set_encoder_train_eval(model, frozen=True)
        return True

    # Unfreeze
    _set_encoder_requires_grad(cfg, model, trainable=True, keep_moe_trainable=False)
    _set_encoder_train_eval(model, frozen=False)
    return False


def train_one_epoch(
    *,
    cfg=None,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    fusion_method: str = "concat",
    f1_average: str = "macro",
    step_print_moe: Optional[float] = 100,
    use_amp: bool = True,
    amp_dtype: str = "fp16",
    scaler: Optional[GradScaler] = None,
    max_grad_norm: Optional[float] = None,
    epoch_idx: Optional[int] = None,
) -> Dict[str, float]:
    model.train()

    moe = bool(getattr(model, "_collect_aux_loss", False))
    hag_mode = cfg is not None and str(getattr(cfg, "mode", "")).strip() == "HAGMoE"
    # Debug in main process to avoid num_workers dataset isolation.
    debug_aspect_span = bool(getattr(cfg, "debug_aspect_span", False))

    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None and hasattr(dataset, "reset_match_stats"):
        dataset.reset_match_stats()

    counters: Dict[str, Any] = {
        "total_loss_sum": 0.0,
        "main_loss_sum": 0.0,
        "aux_loss_sum": 0.0,
        "lambda_loss_sum": 0.0,
        "n_steps": 0,
        "all_preds": [],
        "all_labels": [],
    }

    amp_dtype_torch = (
        torch.float16 if (amp_dtype or "").lower().strip() == "fp16" else torch.bfloat16
    )
    step_print_i = int(step_print_moe) if step_print_moe is not None else 0

    hag_log_sums, hag_log_counts = _init_hag_log_buffers()
    aspect_counters = _init_aspect_span_counters()

    routing_agg = RoutingEpochAggregator()
    moe_enabled = bool(getattr(cfg, "_schema_cfg", {}).get("moe", {}).get("enabled", False)) or moe
    moe_acc = (
        MoEMetricsAccumulator(num_labels=getattr(cfg, "num_labels", None), compute_mi=False)
        if moe_enabled
        else None
    )

    for step, batch in enumerate(dataloader):
        batch = _move_batch_to_device(batch)

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            "cuda",
            enabled=bool(use_amp),
            dtype=amp_dtype_torch,
        ):
            outputs = _forward_step(cfg, model, batch, fusion_method)
            logits, extras = _normalize_model_output(outputs)

            loss_total = extras.get("loss", None)
            if loss_total is None:
                loss_total = extras.get("loss_total", None)
            if loss_total is None:
                raise ValueError("Model output missing loss for training")

            # only meaningful for ver2-return path; safe even if missing
            loss_main = extras.get("loss_main", None)
            loss_lambda = extras.get("loss_lambda", None)
            loss_aux = extras.get("aux_loss", None)

        routing_agg.update(extras, model)
        if moe_acc is not None and extras.get("moe_stats") is not None:
            moe_acc.update(extras.get("moe_stats"), batch.get("label"))

        if hag_mode:
            if step == 0:
                print(f"[HAGMoE] output keys: {sorted(list(extras.keys()))}")
                effective = getattr(model, "_last_fusion_method", None)
                model_cfg_fusion = str(getattr(cfg, "hag_fusion_method", "")).strip()
                model_attr_fusion = str(getattr(model, "hag_fusion_method", "")).strip()
                print(
                    "[HAGMoE] "
                f"benchmark_method={fusion_method} "
                f"cfg.hag_fusion_method={model_cfg_fusion or '""'} "
                f"model.hag_fusion_method={model_attr_fusion or '""'} "
                f"effective_fusion={effective}"
                )
            _update_hag_log_buffers(extras, hag_log_sums, hag_log_counts)

        _backward_step(
            loss_total=loss_total,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
        )

        _accumulate_basic_metrics(
            counters=counters,
            loss_total=loss_total,
            loss_main=loss_main,
            loss_lambda=loss_lambda,
            loss_aux=loss_aux,
            moe=moe,
            logits=logits,
            labels=batch["label"],
        )

        if debug_aspect_span:
            _aspect_span_update(
                batch, cfg, aspect_counters, verbose=False, use_cfg_max_len=True
            )

        if (not hag_mode) and step_print_i and (step > 0) and (step % step_print_i == 0):
            if hasattr(model, "print_moe_debug") and callable(getattr(model, "print_moe_debug")):
                try:
                    model.print_moe_debug(topn=3)
                except Exception as e:
                    print("Cannot print_moe_debug:", e)

    denom = max(1, counters["n_steps"])
    acc = float(accuracy_score(counters["all_labels"], counters["all_preds"]))
    f1 = float(f1_score(counters["all_labels"], counters["all_preds"], average=f1_average))
    cm = confusion_matrix(counters["all_labels"], counters["all_preds"])
    cm_norm = (
        cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)
        if cm.size
        else cm
    )

    if hag_mode:
        _print_hag_epoch_loss_summary(
            model=model, hag_log_sums=hag_log_sums, hag_log_counts=hag_log_counts
        )
        if hasattr(model, "print_moe_debug") and callable(getattr(model, "print_moe_debug")):
            try:
                model.print_moe_debug(topn=3)
            except Exception as e:
                print("Cannot print_moe_debug:", e)

    if debug_aspect_span:
        print(
            f"[AspectSpanDiag] split=train total={aspect_counters['match_total']} "
            f"matched={aspect_counters['match_matched']} "
            f"match_rate={(aspect_counters['match_matched'] / max(1, aspect_counters['match_total'])) * 100:.2f}% "
            f"token_mismatch={aspect_counters['token_mismatch_count']} "
            f"truncated={aspect_counters['truncated_count']} "
            f"not_found_raw={aspect_counters['not_found_raw_count']} "
            f"unknown={aspect_counters['unknown_count']} "
            f"avg_mask_sum={(aspect_counters['match_mask_sum'] / max(1, aspect_counters['match_matched'])):.2f}"
        )

    routing_metrics = routing_agg.finalize()
    if routing_metrics is not None:
        _print_routing_epoch_summary(routing_metrics, "train")

    moe_metrics = moe_acc.finalize() if moe_acc is not None else None

    if moe:
        return {
            "loss_total": counters["total_loss_sum"] / denom,
            "loss_main": counters["main_loss_sum"] / denom,
            "loss_lambda": counters["lambda_loss_sum"] / denom,
            "aux_loss": counters["aux_loss_sum"] / denom,
            "acc": acc,
            "f1": f1,
            "routing": routing_metrics,
            "moe_metrics": moe_metrics,
            "confusion": {"cm": cm.tolist(), "cm_normalized": cm_norm.tolist() if cm.size else []},
            "all_labels": counters["all_labels"],
            "all_preds": counters["all_preds"],
        }

    return {
        "loss": counters["total_loss_sum"] / denom,
        "acc": acc,
        "f1": f1,
        "routing": routing_metrics,
        "moe_metrics": moe_metrics,
        "confusion": {"cm": cm.tolist(), "cm_normalized": cm_norm.tolist() if cm.size else []},
        "all_labels": counters["all_labels"],
        "all_preds": counters["all_preds"],
    }


def eval_model(
    *,
    cfg=None,
    model: nn.Module,
    dataloader: DataLoader,
    id2label: Optional[Dict[int, str]] = None,
    verbose_report: bool = False,
    print_cf_matrix: bool = True,
    fusion_method: str = "concat",
    f1_average: str = "macro",
    return_confusion: bool = False,
    epoch_idx: Optional[int] = None,
    split: str = "eval",
    debug_aspect_span: bool = False,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_logits: list[np.ndarray] = []

    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None and hasattr(dataset, "reset_match_stats"):
        dataset.reset_match_stats()

    debug_aspect_span = bool(debug_aspect_span)
    aspect_counters = _init_aspect_span_counters()
    moe_enabled = bool(getattr(cfg, "_schema_cfg", {}).get("moe", {}).get("enabled", False))
    moe_acc = (
        MoEMetricsAccumulator(
            num_labels=len(id2label) if id2label is not None else None,
            compute_mi=split in {"val", "test"},
        )
        if moe_enabled
        else None
    )
    routing_agg = RoutingEpochAggregator()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = _move_batch_to_device(batch)

            hag_mode = cfg is not None and str(getattr(cfg, "mode", "")).strip() == "HAGMoE"
            outputs = _forward_step(cfg, model, batch, fusion_method)
            logits, extras = _normalize_model_output(outputs)

            loss = extras.get("loss", None)
            if loss is None:
                loss = extras.get("loss_total", None)

            routing_agg.update(extras, model)
            if moe_acc is not None and extras.get("moe_stats") is not None:
                moe_acc.update(extras.get("moe_stats"), batch.get("label"))

            if hag_mode and batch_idx == 0:
                effective = getattr(model, "_last_fusion_method", None)
                model_cfg_fusion = str(getattr(cfg, "hag_fusion_method", "")).strip()
                model_attr_fusion = str(getattr(model, "hag_fusion_method", "")).strip()
                print(
                    "[HAGMoE] "
                    f"benchmark_method={fusion_method} "
                    f"cfg.hag_fusion_method={model_cfg_fusion or '""'} "
                    f"model.hag_fusion_method={model_attr_fusion or '""'} "
                    f"effective_fusion={effective}"
                )

            if loss is not None:
                total_loss += float(loss.item())

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())
            all_logits.append(logits.detach().cpu().numpy())

            _aspect_span_update(
                batch,
                cfg,
                aspect_counters,
                verbose=debug_aspect_span,
                use_cfg_max_len=False,
                epoch_idx=epoch_idx,
                split=split,
                batch_idx=batch_idx,
            )

    avg_loss = total_loss / max(1, len(dataloader))
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=f1_average)

    if verbose_report and id2label is not None:
        target_names = [id2label[i] for i in range(len(id2label))]
        print("Classification report:")
        print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    num_labels = len(id2label) if id2label is not None else None
    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(num_labels)) if num_labels is not None else None,
    )

    if print_cf_matrix:
        print_confusion_matrix(all_labels, all_preds, id2label=id2label, normalize=True)

    f1_per_class = f1_score(all_labels, all_preds, average=None)
    out: Dict[str, Any] = {"loss": avg_loss, "acc": acc, "f1": f1, "f1_per_class": f1_per_class}
    if return_confusion:
        out["confusion"] = cm  # raw counts [C, C]

    if debug_aspect_span:
        print(
            f"[AspectSpanDiag] split={split} total={aspect_counters['match_total']} "
            f"matched={aspect_counters['match_matched']} "
            f"match_rate={(aspect_counters['match_matched'] / max(1, aspect_counters['match_total'])) * 100:.2f}% "
            f"token_mismatch={aspect_counters['token_mismatch_count']} "
            f"truncated={aspect_counters['truncated_count']} "
            f"not_found_raw={aspect_counters['not_found_raw_count']} "
            f"unknown={aspect_counters['unknown_count']} "
            f"avg_mask_sum={(aspect_counters['match_mask_sum'] / max(1, aspect_counters['match_matched'])):.2f}"
        )

    routing_metrics = routing_agg.finalize()
    if routing_metrics is not None and split in {"train", "val", "test"}:
        _print_routing_epoch_summary(routing_metrics, split)

    moe_metrics = moe_acc.finalize() if moe_acc is not None else None
    out["routing"] = routing_metrics
    out["moe_metrics"] = moe_metrics
    if all_logits:
        logits_np = np.concatenate(all_logits, axis=0)
        labels_np = np.asarray(all_labels, dtype=np.int64)
        out["calibration"] = compute_calibration(logits_np, labels_np)
    return out


def run_training_loop(
    cfg,
    model,
    method,
    train_loader,
    val_loader,
    test_loader,
    id2label,
    tag,
):
    moe = bool(getattr(model, "_collect_aux_loss", False))

    if moe:
        history = {
            "train_total_loss": [],
            "train_main_loss": [],
            "train_lambda_loss": [],
            "train_f1": [],
            "train_acc": [],
            "val_loss": [],
            "val_f1": [],
        }
    else:
        history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    best_macro_f1 = -1.0
    best_state_dict = None
    best_epoch = -1
    epochs_no_improve = 0
    last_val_metrics = None
    last_test_metrics = None
    last_train_metrics = None

    print("=======================================================================")
    print("Fusion Method:", method)

    steps_per_epoch = max(1, len(train_loader))

    def trainable_params():
        return [p for p in model.parameters() if p.requires_grad]

    seed_str, fold_str = _extract_seed_fold(tag, cfg)
    try:
        seed_val = int(seed_str)
    except Exception:
        seed_val = seed_str
    try:
        fold_val = int(fold_str)
    except Exception:
        fold_val = fold_str

    # Freeze phases:
    # - Epoch 0 warmup uses freeze0 result before optimizer/scheduler build.
    # - Unfreeze boundary rebuilds optimizer to include newly trainable params.
    # Phase fix: apply freeze for epoch 0 BEFORE building optimizer/scheduler
    freeze0 = maybe_freeze_encoder(cfg, model, epoch_idx_0based=0)

    warmup_ratio_phase1 = 0.0 if freeze0 else float(cfg.warmup_ratio)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        lr=cfg.lr,
        lr_head=cfg.lr_head,
        warmup_ratio=warmup_ratio_phase1,
        total_steps=steps_per_epoch * max(1, int(cfg.epochs)),
        params=trainable_params(),
        adamw_foreach=cfg.adamw_foreach,
        adamw_fused=cfg.adamw_fused,
    )


    scaler = GradScaler() if cfg.use_amp else None
            

    for epoch in range(int(cfg.epochs)):
        print("=======================================================================")
        print(f"{tag}Epoch {epoch + 1}/{cfg.epochs}")

        # Apply freeze policy for this epoch
        if epoch == 0:
            freeze = freeze0
        else:
            freeze = maybe_freeze_encoder(cfg, model, epoch_idx_0based=epoch)

        if cfg.freeze_epochs > 0 and epoch < cfg.freeze_epochs:
            print(f"Encoder frozen (epoch {epoch + 1}/{cfg.freeze_epochs})")
        enc_summary = _encoder_grad_summary(model)
        if enc_summary is not None:
            print(
                f"Encoder params trainable: {enc_summary['trainable']}/{enc_summary['total']} "
                f"(frozen={enc_summary['frozen']})"
            )
        if cfg.freeze_epochs > 0 and epoch == cfg.freeze_epochs:
            best_macro_f1 = -1.0
            best_state_dict = None
            best_epoch = -1
            epochs_no_improve = 0
            print("Unfreeze stage starts: reset best checkpoint and patience.")

        if cfg is not None and str(getattr(cfg, "mode", "")).strip() == "HAGMoE":
            if hasattr(model, "maybe_update_group_temperature"):
                try:
                    model.maybe_update_group_temperature(
                        epoch_idx=epoch, total_epochs=int(cfg.epochs)
                    )
                except Exception as e:
                    print(f"[HAGMoE] cannot update group temperature: {e}")
            _apply_hagmoe_router_schedules(cfg, model, epoch)

        # Rebuild optimizer exactly at unfreeze boundary
        if (cfg.freeze_epochs > 0) and (epoch == cfg.freeze_epochs):
            print("Encoder unfrozen, rebuilding optimizer to include newly-trainable params")
            try:
                del optimizer
                del scheduler
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            remaining_steps = steps_per_epoch * max(1, int(cfg.epochs) - int(epoch))

            # Phase 2 uses the real warmup_ratio (or set to 0.0 if you want to avoid reset completely)
            warmup_ratio_phase2 = 0.0

            optimizer, scheduler = build_optimizer_and_scheduler(
                model=model,
                lr=cfg.lr,
                lr_head=cfg.lr_head,
                warmup_ratio=warmup_ratio_phase2,
                total_steps=remaining_steps,
                params=trainable_params(),
                adamw_foreach=cfg.adamw_foreach,
                adamw_fused=cfg.adamw_fused,
            )

        # Training
        train_metrics = train_one_epoch(
            cfg=cfg,
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            fusion_method=method,
            f1_average="macro",
            step_print_moe=cfg.step_print_moe,
            use_amp=cfg.use_amp,
            amp_dtype=cfg.amp_dtype,
            scaler=scaler,
            max_grad_norm=cfg.max_grad_norm,
            epoch_idx=epoch,
        )
        last_train_metrics = train_metrics

        if "all_labels" in train_metrics and "all_preds" in train_metrics:
            print("Train Confusion Matrix")
            print_confusion_matrix(
                train_metrics["all_labels"],
                train_metrics["all_preds"],
                id2label=id2label,
                normalize=True,
            )
            train_metrics.pop("all_labels", None)
            train_metrics.pop("all_preds", None)

        if moe:
            history["train_total_loss"].append(train_metrics["loss_total"])
            history["train_main_loss"].append(train_metrics["loss_main"])
            history["train_lambda_loss"].append(train_metrics["loss_lambda"])
            history["train_f1"].append(train_metrics["f1"])
            history["train_acc"].append(train_metrics["acc"])

            _write_routing_entry(
                cfg=cfg,
                tag=tag,
                method=method,
                split="train",
                routing_metrics=train_metrics.get("routing"),
                epoch_idx=epoch,
                loss=train_metrics.get("loss_total"),
                macro_f1=train_metrics.get("f1"),
            )

            log = (
                f"Train main_loss {train_metrics['loss_main']:.6f} "
                f"aux_loss {train_metrics['aux_loss']:.6f} "
                f"lambda_loss {train_metrics['loss_lambda']:.6f} "
                f"total_loss {train_metrics['loss_total']:.6f} "
                f"\nTrain F1 {train_metrics['f1']:.4f} acc {train_metrics['acc']:.4f}"
            )
            log += "\n"
        else:
            history["train_loss"].append(float(train_metrics["loss"]))
            history["train_f1"].append(float(train_metrics["f1"]))

            _write_routing_entry(
                cfg=cfg,
                tag=tag,
                method=method,
                split="train",
                routing_metrics=train_metrics.get("routing"),
                epoch_idx=epoch,
                loss=train_metrics.get("loss"),
                macro_f1=train_metrics.get("f1"),
            )

            log = (
                f"Train loss {train_metrics['loss']:.4f} "
                f"F1 {train_metrics['f1']:.4f} "
                f"acc {train_metrics['acc']:.4f}"
            )
            log += "\n"


        if val_loader is not None:
            print("Validation Confusion Matrix")
            val_metrics = eval_model(
                cfg=cfg,
                model=model,
                dataloader=val_loader,
                id2label=id2label,
                print_cf_matrix=True,
                verbose_report=False,
                fusion_method=method,
                f1_average="macro",
                epoch_idx=epoch,
                split="val",
                debug_aspect_span=getattr(cfg, "debug_aspect_span", False),
                return_confusion=True,
            )
            last_val_metrics = val_metrics
            history["val_loss"].append(float(val_metrics["loss"]))
            history["val_f1"].append(float(val_metrics["f1"]))

            macro_f1 = float(val_metrics["f1"])

            _write_routing_entry(
                cfg=cfg,
                tag=tag,
                method=method,
                split="val",
                routing_metrics=val_metrics.get("routing"),
                epoch_idx=epoch,
                loss=val_metrics.get("loss"),
                macro_f1=macro_f1,
            )

            log += (
                f"Val loss {val_metrics['loss']:.4f} "
                f"F1 {val_metrics['f1']:.4f} "
                f"acc {val_metrics['acc']:.4f} "
            )

            if val_metrics.get("confusion") is not None:
                cm = np.asarray(val_metrics["confusion"], dtype=np.float64)
                cm_norm = (cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)).tolist()
                conf_block = {"cm": cm.tolist(), "cm_normalized": cm_norm}
            else:
                conf_block = None

            should_save = (macro_f1 > best_macro_f1)
            if epoch < cfg.freeze_epochs:
                print("Skip saving best model during freeze stage.")
            elif should_save:
                best_macro_f1 = macro_f1
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_no_improve = 0
                print()
                print("*"*100)
                print("[MODEL] New best model on macro_f1")
                print()
            elif epoch >= cfg.freeze_epochs:
                epochs_no_improve += 1
                if cfg.early_stop_patience > 0 and epochs_no_improve >= int(cfg.early_stop_patience):
                    print(f"Early stopping triggered after {cfg.early_stop_patience} epochs without improvement")
                    print(log)
                    break

        if test_loader is not None:
            print("Test Confusion Matrix")
            test_metrics = eval_model(
                cfg=cfg,
                model=model,
                dataloader=test_loader,
                id2label=id2label,
                print_cf_matrix=True,
                verbose_report=False,
                fusion_method=method,
                f1_average="macro",
                epoch_idx=epoch,
                split="test",
                debug_aspect_span=getattr(cfg, "debug_aspect_span", False),
                return_confusion=True,
            )
            last_test_metrics = test_metrics
            _write_routing_entry(
                cfg=cfg,
                tag=tag,
                method=method,
                split="test",
                routing_metrics=test_metrics.get("routing"),
                epoch_idx=epoch,
                loss=test_metrics.get("loss"),
                macro_f1=test_metrics.get("f1"),
            )
            log += (
                f"\nTest loss {test_metrics['loss']:.4f} "
                f"F1 {test_metrics['f1']:.4f} "
                f"acc {test_metrics['acc']:.4f} "
            )

            if test_metrics.get("confusion") is not None:
                cm = np.asarray(test_metrics["confusion"], dtype=np.float64)
                cm_norm = (cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)).tolist()
                conf_block = {"cm": cm.tolist(), "cm_normalized": cm_norm}
            else:
                conf_block = None

            if val_loader is None:
                macro_f1 = float(test_metrics["f1"])
                should_save = (macro_f1 > best_macro_f1)
                if epoch < cfg.freeze_epochs:
                    print("Skip saving best model during freeze stage.")
                elif should_save:
                    best_macro_f1 = macro_f1
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    epochs_no_improve = 0
                    print()
                    print("*"*100)
                    print("[MODEL] New best model on macro_f1 (test-based)")
                    print()
                elif epoch >= cfg.freeze_epochs:
                    epochs_no_improve += 1
                    if cfg.early_stop_patience > 0 and epochs_no_improve >= int(cfg.early_stop_patience):
                        print(f"Early stopping triggered after {cfg.early_stop_patience} epochs without improvement")
                        print(log)
                        break

        print(log)

    try:
        del optimizer
        del scheduler
        del scaler
    except Exception:
        pass

    cleanup_cuda()

    return {
        "best_state_dict": best_state_dict,
        "best_epoch": best_epoch,
        "history": history,
        "last_val_metrics": last_val_metrics,
        "last_test_metrics": last_test_metrics,
        "last_train_metrics": last_train_metrics,
    }
