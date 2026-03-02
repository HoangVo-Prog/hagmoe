from __future__ import annotations

import gc
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .const import DEVICE


def cleanup_cuda(*objs: Any) -> None:
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def mean_std(xs: list[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def aggregate_confusions(cms: list[np.ndarray]) -> dict:
    if len(cms) == 0:
        return {}

    arr = np.stack([np.asarray(cm, dtype=np.float64) for cm in cms], axis=0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)

    denom = mean.sum(axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    mean_norm = mean / denom
    std_norm = std / denom

    return {
        "cm_mean": mean.tolist(),
        "cm_std": std.tolist(),
        "cm_mean_normalized": mean_norm.tolist(),
        "cm_std_normalized": std_norm.tolist(),
    }


def safe_float(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (float, int)):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().item())
    return float("nan")


def logits_to_metrics(logits: np.ndarray, labels: np.ndarray):
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"acc": float(acc), "f1": float(f1)}


@torch.no_grad()
def collect_test_logits(
    *,
    model: torch.nn.Module,
    test_loader,
    fusion_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_chunks = []
    labels_chunks = []

    for batch in test_loader:
        batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
        model_kwargs = {
            "input_ids_sent": batch["input_ids_sent"],
            "attention_mask_sent": batch["attention_mask_sent"],
            "input_ids_term": batch["input_ids_term"],
            "attention_mask_term": batch["attention_mask_term"],
            "labels": None,
            "fusion_method": fusion_method,
        }
        if "aspect_start" in batch:
            model_kwargs["aspect_start"] = batch.get("aspect_start")
        if "aspect_end" in batch:
            model_kwargs["aspect_end"] = batch.get("aspect_end")
        if "aspect_mask_sent" in batch:
            model_kwargs["aspect_mask_sent"] = batch.get("aspect_mask_sent")

        outputs = model(**model_kwargs)
        logits = outputs["logits"].detach().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy()

        logits_chunks.append(logits)
        labels_chunks.append(labels)

    logits_all = np.concatenate(logits_chunks, axis=0)
    labels_all = np.concatenate(labels_chunks, axis=0)
    return logits_all, labels_all
    
