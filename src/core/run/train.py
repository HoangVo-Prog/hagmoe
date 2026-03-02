import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix

from src.core.utils.helper import (
    set_seed,
    get_tokenizer,
    get_dataset,
    get_dataloader,
)
from .engine import run_training_loop
from .model_factory import build_model
from src.core.utils.general import (
    cleanup_cuda,
    collect_test_logits,
    logits_to_metrics,
    mean_std,
    aggregate_confusions,
)
from src.core.utils.plotting import print_confusion_matrix
from src.core.utils.artifacts import save_artifacts, aggregate_metrics


def run_single_train_eval(config, method=None):
    os.makedirs(config.output_dir, exist_ok=True)
    tokenizer = get_tokenizer(config)
    train_set, test_set = get_dataset(config, tokenizer)
    config.label2id = train_set.label2id
    config.id2label = {v: k for k, v in train_set.label2id.items()}

    train_loader, _, test_loader = get_dataloader(
        cfg=config, train_set=train_set, val_set=None, test_set=test_set
    )

    seeds = (
        list(getattr(config, "seed_list", None) or [])
        or [int(config.seed) + i for i in range(int(config.num_seeds))]
    )

    raw_methods = []
    raw_value = getattr(config, "benchmark_methods", "")
    if isinstance(raw_value, (list, tuple)):
        raw_methods = [str(m).strip() for m in raw_value if str(m).strip()]
    else:
        raw = str(raw_value or "").strip()
        if raw:
            raw_methods = [m.strip() for m in raw.split(",") if m.strip()]
    if not raw_methods:
        chosen = method or getattr(config, "fusion_method", None) or "default"
        raw_methods = [chosen]

    num_classes = len(config.label2id)
    original_fusion = getattr(config, "fusion_method", None)
    method_summaries = {}
    
    print(f"Running {raw_methods}:")
    if not raw_methods:
        raise ValueError("No benchmark methods resolved; check benchmark_methods input.")

    def _run_for_method(fusion_method: str) -> dict:
        config.fusion_method = fusion_method
        per_seed_records = []
        all_seed_logits = []
        all_seed_cms = []
        labels_last = None

        for seed in seeds:
            print(f"\n===== SINGLE seed={seed} fusion={fusion_method} =====")
            set_seed(int(seed))

            model = build_model(config)
            out = run_training_loop(
                cfg=config,
                model=model,
                method=fusion_method,
                train_loader=train_loader,
                val_loader=None,
                test_loader=test_loader,
                id2label=config.id2label,
                tag=f"[SINGLE seed={seed}] ",
            )

            if out.get("best_state_dict") is not None:
                model.load_state_dict(out["best_state_dict"])
                if out.get("best_epoch") is not None:
                    print(f"Loaded best SINGLE model from epoch {out.get('best_epoch')}")

            train_logits, train_labels = collect_test_logits(
                model=model,
                test_loader=train_loader,
                fusion_method=fusion_method,
            )
            train_metrics = logits_to_metrics(train_logits, train_labels)
            train_preds = train_logits.argmax(axis=-1)
            train_cm = confusion_matrix(train_labels, train_preds, labels=list(range(num_classes)))

            print("\n" + "-" * 100)
            print(f"[Seed {seed}] Best epoch: {int(out.get('best_epoch')) + 1}")
            print(
                f"Train acc: {train_metrics['acc']:.4f} | Train f1: {train_metrics['f1']:.4f}"
            )
            print("Train Confusion Matrix (normalized):")
            print_confusion_matrix(
                train_labels.tolist(),
                train_preds.tolist(),
                id2label=config.id2label,
                normalize=True,
            )
            print("-" * 100)

            logits, labels = collect_test_logits(
                model=model,
                test_loader=test_loader,
                fusion_method=fusion_method,
            )
            labels_last = labels
            metrics = logits_to_metrics(logits, labels)
            preds = logits.argmax(axis=-1)
            cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

            print(f"Test acc: {metrics['acc']:.4f} | Test f1: {metrics['f1']:.4f}")
            print("Test Confusion Matrix (normalized):")
            print_confusion_matrix(
                labels.tolist(),
                preds.tolist(),
                id2label=config.id2label,
                normalize=True,
            )
            all_seed_cms.append(cm)

            extra = out.get("last_test_metrics") or {}
            f1_per_class = extra.get("f1_per_class")
            if hasattr(f1_per_class, "tolist"):
                f1_per_class = f1_per_class.tolist()

            metrics_block = {
                "loss": extra.get("loss"),
                "acc": float(metrics.get("acc", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                "f1_per_class": f1_per_class,
                "confusion": {
                    "cm": cm.tolist(),
                    "cm_normalized": (
                        cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)
                    ).tolist(),
                },
                "moe_metrics": extra.get("moe_metrics"),
                "calibration": extra.get("calibration"),
            }

            save_artifacts(
                output_dir=config.output_dir,
                mode=config.mode,
                method=fusion_method,
                loss_type=config.loss_type,
                seed=int(seed),
                fold="full",
                split="test",
                metrics=metrics_block,
            )

            per_seed_records.append(
                {
                    "seed": int(seed),
                    "train_acc": float(train_metrics.get("acc", 0.0)),
                    "train_f1": float(train_metrics.get("f1", 0.0)),
                    "train_confusion": {
                        "cm": train_cm.tolist(),
                        "cm_normalized": (
                            train_cm
                            / np.clip(train_cm.sum(axis=1, keepdims=True), 1e-12, None)
                        ).tolist(),
                    },
                    "acc": float(metrics.get("acc", 0.0)),
                    "f1": float(metrics.get("f1", 0.0)),
                    "f1_per_class": f1_per_class,
                    "confusion": metrics_block["confusion"],
                    "calibration": metrics_block.get("calibration"),
                    "moe_metrics": metrics_block.get("moe_metrics"),
                }
            )
            all_seed_logits.append(logits)

            del model
            cleanup_cuda()
            
        accs = [float(r["acc"]) for r in per_seed_records]
        f1s = [float(r["f1"]) for r in per_seed_records]
        acc_mean, acc_std = mean_std(accs)
        f1_mean, f1_std = mean_std(f1s)
        acc_min = float(np.min(accs)) if accs else float("nan")
        acc_max = float(np.max(accs)) if accs else float("nan")
        f1_min = float(np.min(f1s)) if f1s else float("nan")
        f1_max = float(np.max(f1s)) if f1s else float("nan")

        agg_confusions = aggregate_confusions(all_seed_cms)
        metrics_list = []
        for record in per_seed_records:
            metrics_list.append(
                {
                    "loss": None,
                    "acc": float(record.get("acc", 0.0)),
                    "f1": float(record.get("f1", 0.0)),
                    "f1_per_class": record.get("f1_per_class"),
                    "confusion": record.get("confusion"),
                    "moe_metrics": record.get("moe_metrics"),
                    "calibration": record.get("calibration"),
                }
            )
        agg = aggregate_metrics(metrics_list)
        if agg is not None:
            save_artifacts(
                output_dir=config.output_dir,
                mode=config.mode,
                method=fusion_method,
                loss_type=config.loss_type,
                seed="avg",
                fold="full",
                split="test",
                metrics=agg,
            )

        ensemble_block = None
        if getattr(config, "do_ensemble_logits", False) and len(all_seed_logits) >= 2:
            ens_logits = np.mean(np.stack(all_seed_logits, axis=0), axis=0)
            if labels_last is None:
                raise RuntimeError("labels not collected for ensemble")

            ens_metrics = logits_to_metrics(ens_logits, labels_last)
            ens_preds = ens_logits.argmax(axis=-1)
            ens_cm = confusion_matrix(labels_last, ens_preds, labels=list(range(num_classes)))
            ens_metrics_block = {
                "loss": None,
                "acc": float(ens_metrics.get("acc", 0.0)),
                "f1": float(ens_metrics.get("f1", 0.0)),
                "confusion": {
                    "cm": ens_cm.tolist(),
                    "cm_normalized": (
                        ens_cm / np.clip(ens_cm.sum(axis=1, keepdims=True), 1e-12, None)
                    ).tolist(),
                },
            }
            save_artifacts(
                output_dir=config.output_dir,
                mode=config.mode,
                method=fusion_method,
                loss_type=config.loss_type,
                seed="ens",
                fold="full",
                split="test",
                metrics=ens_metrics_block,
            )
            ensemble_block = {
                "metrics": ens_metrics_block,
            }

        return {
            "mode": config.mode,
            "method": fusion_method,
            "loss_type": config.loss_type,
            "seeds": [int(s) for s in seeds],
            "runs": per_seed_records,
            "summary": {
                "acc_mean": float(acc_mean),
                "acc_std": float(acc_std),
                "acc_min": acc_min,
                "acc_max": acc_max,
                "f1_mean": float(f1_mean),
                "f1_std": float(f1_std),
                "f1_min": f1_min,
                "f1_max": f1_max,
            },
            "confusion": agg_confusions,
            "ensemble": ensemble_block,
        }

    for fusion_method in raw_methods:
        method_summaries[fusion_method] = _run_for_method(fusion_method)

    if original_fusion is not None:
        config.fusion_method = original_fusion

    summary_path = os.path.join(config.output_dir, config.output_name)
    if len(raw_methods) == 1:
        summary = method_summaries[raw_methods[0]]
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)
        return summary

    combined = {
        "mode": config.mode,
        "loss_type": config.loss_type,
        "methods": method_summaries,
        "method_order": list(raw_methods),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=True, indent=2)

    return combined
        
