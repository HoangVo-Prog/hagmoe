from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_artifacts(
    *,
    output_dir: str,
    mode: str,
    method: str,
    loss_type: str,
    seed: int | str,
    fold: int | str,
    split: str,
    metrics: Dict[str, Any],
) -> None:
    base = os.path.join(
        output_dir,
        str(mode),
        str(method),
        str(loss_type),
        f"seed_{seed}",
        f"fold_{fold}",
        str(split),
    )
    os.makedirs(base, exist_ok=True)

    _write_json(os.path.join(base, "metrics.json"), _to_jsonable(metrics))

    confusion = metrics.get("confusion")
    if confusion is not None:
        _plot_confusion(confusion, os.path.join(base, "confusion.png"))

    calibration = metrics.get("calibration")
    if calibration is not None:
        _plot_reliability(calibration, os.path.join(base, "reliability.png"))
        _plot_confidence_hist(calibration, os.path.join(base, "confidence_hist.png"))

    moe_metrics = metrics.get("moe_metrics")
    if moe_metrics is not None:
        _plot_entropy_hist(moe_metrics, os.path.join(base, "router_entropy_hist.png"))
        _plot_expert_usage(moe_metrics, os.path.join(base, "expert_usage.png"))
        _plot_top1_hist(moe_metrics, os.path.join(base, "top1_hist.png"))


def aggregate_metrics(metrics_list: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    items = [m for m in metrics_list if isinstance(m, dict)]
    if not items:
        return None

    agg: Dict[str, Any] = {}
    for key in ["loss", "acc", "f1"]:
        vals = [m.get(key) for m in items if m.get(key) is not None]
        if vals:
            agg[key] = float(np.mean(vals))

    f1pcs = [m.get("f1_per_class") for m in items if m.get("f1_per_class") is not None]
    if f1pcs:
        arrs = [np.asarray(v, dtype=np.float64) for v in f1pcs]
        agg["f1_per_class"] = np.mean(np.stack(arrs, axis=0), axis=0).tolist()

    confs = [m.get("confusion") for m in items if m.get("confusion") is not None]
    if confs:
        cms = [np.asarray(c.get("cm"), dtype=np.float64) for c in confs if c.get("cm") is not None]
        if cms:
            cm_mean = np.mean(np.stack(cms, axis=0), axis=0)
            cm_norm = (cm_mean / np.clip(cm_mean.sum(axis=1, keepdims=True), 1e-12, None)).tolist()
            agg["confusion"] = {"cm": cm_mean.tolist(), "cm_normalized": cm_norm}

    calibrations = [m.get("calibration") for m in items if m.get("calibration") is not None]
    if calibrations:
        agg["calibration"] = _aggregate_calibration(calibrations)

    moes = [m.get("moe_metrics") for m in items if m.get("moe_metrics") is not None]
    if moes:
        agg["moe_metrics"] = _aggregate_moe_metrics(moes)

    return agg


def _aggregate_calibration(items: list[Dict[str, Any]]) -> Dict[str, Any]:
    bins = items[0].get("bins", [])
    n_bins = len(bins)
    bin_counts = [0] * n_bins
    bin_acc_sum = [0.0] * n_bins
    bin_conf_sum = [0.0] * n_bins
    n_list = []
    nll_vals = []
    brier_vals = []
    ece_vals = []
    edges = None
    hist_counts = None

    for item in items:
        conf_hist = item.get("conf_hist", {})
        counts = conf_hist.get("counts")
        if counts is not None:
            n_list.append(int(np.sum(counts)))
        if item.get("nll") is not None:
            nll_vals.append(float(item["nll"]))
        if item.get("brier") is not None:
            brier_vals.append(float(item["brier"]))
        if item.get("ece") is not None:
            ece_vals.append(float(item["ece"]))

        if edges is None:
            edges = conf_hist.get("edges")
        if hist_counts is None and counts is not None:
            hist_counts = np.zeros_like(np.asarray(counts, dtype=np.int64))
        if counts is not None and hist_counts is not None:
            hist_counts += np.asarray(counts, dtype=np.int64)

        bins_item = item.get("bins", [])
        for i, b in enumerate(bins_item):
            cnt = b.get("count") or 0
            if cnt <= 0:
                continue
            bin_counts[i] += int(cnt)
            if b.get("acc") is not None:
                bin_acc_sum[i] += float(b["acc"]) * cnt
            if b.get("conf") is not None:
                bin_conf_sum[i] += float(b["conf"]) * cnt

    bins_out = []
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bins_out.append(
                {
                    "bin": i,
                    "acc": float(bin_acc_sum[i] / bin_counts[i]),
                    "conf": float(bin_conf_sum[i] / bin_counts[i]),
                    "count": int(bin_counts[i]),
                }
            )
        else:
            bins_out.append({"bin": i, "acc": None, "conf": None, "count": 0})

    n_weights = np.asarray(n_list, dtype=np.float64)
    total_n = float(n_weights.sum()) if n_weights.size else 0.0
    def _wavg(vals):
        if not vals:
            return None
        if total_n > 0 and len(vals) == len(n_list):
            return float(np.sum(np.asarray(vals) * n_weights) / total_n)
        return float(np.mean(vals))

    return {
        "nll": _wavg(nll_vals),
        "brier": _wavg(brier_vals),
        "ece": _wavg(ece_vals),
        "bins": bins_out,
        "conf_hist": {
            "edges": edges if edges is not None else [],
            "counts": hist_counts.tolist() if hist_counts is not None else [],
        },
    }


def _aggregate_moe_metrics(items: list[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}
    for key in [
        "entropy_norm_mean",
        "entropy_norm_std",
        "kl_to_uniform_mean",
        "margin_mean",
        "effective_num_experts",
        "dead_count",
        "uniform_rate",
        "mi_top1_label",
        "entropy_norm_mean_raw",
        "entropy_norm_std_raw",
        "kl_to_uniform_mean_raw",
        "margin_mean_raw",
        "effective_num_experts_raw",
        "dead_count_raw",
        "uniform_rate_raw",
        "mi_top1_label_raw",
    ]:
        vals = [m.get(key) for m in items if m.get(key) is not None]
        if vals:
            agg[key] = float(np.mean(vals))

    for key in ["mean_prob", "top1_hist", "entropy_hist"]:
        vals = [m.get(key) for m in items if m.get(key) is not None]
        if vals:
            arrs = [np.asarray(v, dtype=np.float64) for v in vals]
            agg[key] = np.mean(np.stack(arrs, axis=0), axis=0).tolist()

    per_label_entropy = {}
    per_label_top1 = {}
    for m in items:
        ple = m.get("per_label_entropy_mean") or {}
        for k, v in ple.items():
            per_label_entropy.setdefault(k, []).append(float(v))
        plt = m.get("per_label_top1_hist") or {}
        for k, v in plt.items():
            per_label_top1.setdefault(k, []).append(np.asarray(v, dtype=np.float64))

    if per_label_entropy:
        agg["per_label_entropy_mean"] = {
            k: float(np.mean(v)) for k, v in per_label_entropy.items()
        }
    if per_label_top1:
        agg["per_label_top1_hist"] = {
            k: np.mean(np.stack(v, axis=0), axis=0).tolist() for k, v in per_label_top1.items()
        }

    per_label_top1_raw = {}
    for m in items:
        plt_raw = m.get("per_label_top1_hist_raw") or {}
        for k, v in plt_raw.items():
            per_label_top1_raw.setdefault(k, []).append(np.asarray(v, dtype=np.float64))
    if per_label_top1_raw:
        agg["per_label_top1_hist_raw"] = {
            k: np.mean(np.stack(v, axis=0), axis=0).tolist() for k, v in per_label_top1_raw.items()
        }

    return agg


def _write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _plot_confusion(confusion: Dict[str, Any], path: str) -> None:
    cm = np.asarray(confusion.get("cm", []), dtype=np.float32)
    if cm.size == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_reliability(calibration: Dict[str, Any], path: str) -> None:
    bins = calibration.get("bins", [])
    if not bins:
        return
    acc = [b["acc"] if b["acc"] is not None else np.nan for b in bins]
    conf = [b["conf"] if b["conf"] is not None else np.nan for b in bins]
    x = np.arange(len(acc))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, len(acc) - 1], [0, 1], linestyle="--", color="gray")
    ax.plot(x, acc, marker="o", label="Accuracy")
    ax.plot(x, conf, marker="o", label="Confidence")
    ax.set_title("Reliability Diagram")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_confidence_hist(calibration: Dict[str, Any], path: str) -> None:
    hist = calibration.get("conf_hist", {})
    edges = hist.get("edges")
    counts = hist.get("counts")
    if not edges or not counts:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(len(counts)), counts, width=0.8)
    ax.set_title("Confidence Histogram")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_entropy_hist(moe_metrics: Dict[str, Any], path: str) -> None:
    hist = moe_metrics.get("entropy_hist")
    if not hist:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(len(hist)), hist, width=0.8)
    ax.set_title("Router Entropy Histogram")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Fraction")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_expert_usage(moe_metrics: Dict[str, Any], path: str) -> None:
    mean_prob = moe_metrics.get("mean_prob")
    if not mean_prob:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(mean_prob)), mean_prob, width=0.8)
    ax.set_title("Expert Usage (Mean Prob)")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Mean Prob")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_top1_hist(moe_metrics: Dict[str, Any], path: str) -> None:
    top1_hist = moe_metrics.get("top1_hist")
    if not top1_hist:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(top1_hist)), top1_hist, width=0.8)
    ax.set_title("Top1 Histogram")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Fraction")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
