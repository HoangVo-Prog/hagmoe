from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch


def compute_calibration(
    logits: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    *,
    n_bins: int = 15,
) -> Dict[str, Any]:
    if torch.is_tensor(logits):
        logits = logits.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    probs = _softmax_np(logits)
    labels = labels.astype(np.int64)
    n = probs.shape[0]
    if n == 0:
        return {
            "nll": None,
            "brier": None,
            "ece": None,
            "bins": [],
            "conf_hist": [],
        }

    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float32)

    nll = -np.log(np.clip(probs[np.arange(n), labels], 1e-12, 1.0)).mean()
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    brier = np.mean(np.sum((probs - onehot) ** 2, axis=1))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    bin_stats = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            bin_stats.append({"bin": b, "acc": None, "conf": None, "count": 0})
            continue
        acc_b = float(correct[mask].mean())
        conf_b = float(conf[mask].mean())
        count_b = int(mask.sum())
        ece += (count_b / n) * abs(acc_b - conf_b)
        bin_stats.append({"bin": b, "acc": acc_b, "conf": conf_b, "count": count_b})

    hist_counts, hist_edges = np.histogram(conf, bins=bins)
    conf_hist = {
        "edges": hist_edges.tolist(),
        "counts": hist_counts.tolist(),
    }

    return {
        "nll": float(nll),
        "brier": float(brier),
        "ece": float(ece),
        "bins": bin_stats,
        "conf_hist": conf_hist,
    }


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)
