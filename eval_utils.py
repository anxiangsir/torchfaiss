from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    average_precision_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    precision_recall_curve,
    rand_score,
    roc_auc_score,
    roc_curve,
    v_measure_score,
)


EPS = 1e-12
TOP_CURVE_CLASSES = 5
CONFUSION_TOP_N = 50
CURVE_POINT_LIMIT = 512


@dataclass
class EvaluationArtifacts:
    summary: Dict[str, object]
    paths: Dict[str, str]


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "result"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_div(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    numer_arr = np.asarray(numer, dtype=np.float64)
    denom_arr = np.asarray(denom, dtype=np.float64)
    return np.divide(numer_arr, np.maximum(denom_arr, EPS))


def build_contingency(rows: np.ndarray, cols: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    matrix = np.zeros((n_rows, n_cols), dtype=np.int64)
    np.add.at(matrix, (rows, cols), 1)
    return matrix


def compute_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    n_classes = int(labels_true.max()) + 1
    n_clusters = int(labels_pred.max()) + 1
    contingency = build_contingency(labels_pred, labels_true, n_clusters, n_classes)
    return float(contingency.max(axis=1).sum() / max(len(labels_true), 1))


def compute_clustering_metrics(labels_true: np.ndarray, labels_pred: np.ndarray) -> Dict[str, float]:
    return {
        "nmi": float(normalized_mutual_info_score(labels_true, labels_pred)),
        "ami": float(adjusted_mutual_info_score(labels_true, labels_pred)),
        "ari": float(adjusted_rand_score(labels_true, labels_pred)),
        "rand": float(rand_score(labels_true, labels_pred)),
        "fowlkes_mallows": float(fowlkes_mallows_score(labels_true, labels_pred)),
        "homogeneity": float(homogeneity_score(labels_true, labels_pred)),
        "completeness": float(completeness_score(labels_true, labels_pred)),
        "v_measure": float(v_measure_score(labels_true, labels_pred)),
        "purity": float(compute_purity(labels_true, labels_pred)),
    }


def build_cluster_label_model(
    train_labels: np.ndarray,
    train_assignments: np.ndarray,
    n_classes: int,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cluster_label_counts = build_contingency(train_assignments, train_labels, n_clusters, n_classes)
    cluster_sizes = cluster_label_counts.sum(axis=1)
    global_label_counts = np.bincount(train_labels, minlength=n_classes).astype(np.float64)
    global_probs = global_label_counts / np.maximum(global_label_counts.sum(), EPS)
    smoothed = cluster_label_counts.astype(np.float64) + EPS
    cluster_probs = smoothed / smoothed.sum(axis=1, keepdims=True)
    empty_mask = cluster_sizes == 0
    if np.any(empty_mask):
        cluster_probs[empty_mask] = global_probs
    majority_labels = cluster_label_counts.argmax(axis=1)
    if np.any(empty_mask):
        fallback_label = int(global_probs.argmax())
        majority_labels[empty_mask] = fallback_label
    cluster_purities = _safe_div(cluster_label_counts.max(axis=1), cluster_sizes)
    return cluster_label_counts, cluster_probs, majority_labels, cluster_purities


def compute_classification_metrics(
    confusion: np.ndarray,
    labels: Iterable[int],
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    labels = np.asarray(list(labels), dtype=np.int64)
    tp = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)

    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    total = float(confusion.sum())
    accuracy = float(tp.sum() / max(total, 1.0))
    valid = support > 0
    macro_precision = float(precision[valid].mean()) if valid.any() else 0.0
    macro_recall = float(recall[valid].mean()) if valid.any() else 0.0
    macro_f1 = float(f1[valid].mean()) if valid.any() else 0.0
    weights = _safe_div(support, support.sum())
    weighted_precision = float((precision * weights).sum())
    weighted_recall = float((recall * weights).sum())
    weighted_f1 = float((f1 * weights).sum())
    micro_precision = accuracy
    micro_recall = accuracy
    micro_f1 = accuracy

    per_class: Dict[int, Dict[str, float]] = {}
    for idx, label in enumerate(labels):
        per_class[int(label)] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
            "predicted": int(predicted[idx]),
        }

    summary = {
        "accuracy": accuracy,
        "balanced_accuracy": macro_recall,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
        "precision_weighted": weighted_precision,
        "recall_weighted": weighted_recall,
        "f1_weighted": weighted_f1,
        "precision_micro": micro_precision,
        "recall_micro": micro_recall,
        "f1_micro": micro_f1,
    }
    return summary, per_class


def _weighted_binary_inputs(scores: np.ndarray, positives: np.ndarray, negatives: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    binary_labels = np.concatenate(
        [np.ones(scores.size, dtype=np.int8), np.zeros(scores.size, dtype=np.int8)]
    )
    binary_scores = np.concatenate([scores, scores]).astype(np.float64, copy=False)
    weights = np.concatenate([positives, negatives]).astype(np.float64, copy=False)
    return binary_labels, binary_scores, weights


def compute_multiclass_curve_metrics(
    score_matrix: np.ndarray,
    label_cluster_counts: np.ndarray,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, Dict[str, float]]]:
    if score_matrix.ndim != 2 or score_matrix.shape != (label_cluster_counts.shape[1], label_cluster_counts.shape[0]):
        raise ValueError(
            f"Expected score_matrix shape {(label_cluster_counts.shape[1], label_cluster_counts.shape[0])}, "
            f"got {score_matrix.shape}"
        )

    n_classes, n_clusters = label_cluster_counts.shape
    cluster_totals = label_cluster_counts.sum(axis=0)
    valid_classes = np.where(label_cluster_counts.sum(axis=1) > 0)[0]

    roc_grid = np.linspace(0.0, 1.0, 256)
    pr_grid = np.linspace(0.0, 1.0, 256)
    roc_macro_curves = []
    pr_macro_curves = []
    roc_auc_values = []
    ap_values = []
    per_class: Dict[int, Dict[str, float]] = {}

    support_order = valid_classes[np.argsort(label_cluster_counts.sum(axis=1)[valid_classes])[::-1]]
    top_classes = set(int(x) for x in support_order[:TOP_CURVE_CLASSES])
    top_curve_data: Dict[int, Dict[str, Any]] = {}

    for cls in valid_classes:
        scores = score_matrix[:, cls]
        positives = label_cluster_counts[cls]
        negatives = cluster_totals - positives
        y_true, y_score, sample_weight = _weighted_binary_inputs(scores, positives, negatives)

        if positives.sum() <= 0 or negatives.sum() <= 0:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
        precision, recall, _ = precision_recall_curve(y_true, y_score, sample_weight=sample_weight)
        roc_auc = float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))
        ap = float(average_precision_score(y_true, y_score, sample_weight=sample_weight))

        roc_macro_curves.append(np.interp(roc_grid, fpr, tpr))
        pr_precision_sorted = precision[np.argsort(recall)]
        pr_recall_sorted = np.sort(recall)
        pr_macro_curves.append(np.interp(pr_grid, pr_recall_sorted, pr_precision_sorted))
        roc_auc_values.append(roc_auc)
        ap_values.append(ap)

        per_class[int(cls)] = {
            "roc_auc": roc_auc,
            "average_precision": ap,
        }
        if int(cls) in top_classes:
            top_curve_data[int(cls)] = {
                "roc_fpr": fpr.tolist(),
                "roc_tpr": tpr.tolist(),
                "pr_recall": recall.tolist(),
                "pr_precision": precision.tolist(),
            }

    flat_scores = score_matrix.T.reshape(n_classes * n_clusters)
    flat_positives = label_cluster_counts.reshape(n_classes * n_clusters)
    flat_negatives = (cluster_totals[None, :] - label_cluster_counts).reshape(n_classes * n_clusters)
    y_true_micro, y_score_micro, weights_micro = _weighted_binary_inputs(flat_scores, flat_positives, flat_negatives)
    micro_fpr, micro_tpr, _ = roc_curve(y_true_micro, y_score_micro, sample_weight=weights_micro)
    micro_precision, micro_recall, _ = precision_recall_curve(
        y_true_micro,
        y_score_micro,
        sample_weight=weights_micro,
    )
    micro_fpr_grid = np.linspace(0.0, 1.0, CURVE_POINT_LIMIT)
    micro_recall_grid = np.linspace(0.0, 1.0, CURVE_POINT_LIMIT)
    micro_tpr_grid = np.interp(micro_fpr_grid, micro_fpr, micro_tpr)
    pr_precision_sorted = micro_precision[np.argsort(micro_recall)]
    pr_recall_sorted = np.sort(micro_recall)
    micro_precision_grid = np.interp(micro_recall_grid, pr_recall_sorted, pr_precision_sorted)

    curves = {
        "roc": {
            "micro_fpr": micro_fpr_grid.tolist(),
            "micro_tpr": micro_tpr_grid.tolist(),
            "macro_fpr": roc_grid.tolist(),
            "macro_tpr": np.mean(roc_macro_curves, axis=0).tolist() if roc_macro_curves else [],
            "micro_auc": float(roc_auc_score(y_true_micro, y_score_micro, sample_weight=weights_micro)),
            "macro_auc": float(np.mean(roc_auc_values)) if roc_auc_values else 0.0,
            "top_classes": top_curve_data,
        },
        "pr": {
            "micro_recall": micro_recall_grid.tolist(),
            "micro_precision": micro_precision_grid.tolist(),
            "macro_recall": pr_grid.tolist(),
            "macro_precision": np.mean(pr_macro_curves, axis=0).tolist() if pr_macro_curves else [],
            "micro_ap": float(average_precision_score(y_true_micro, y_score_micro, sample_weight=weights_micro)),
            "macro_ap": float(np.mean(ap_values)) if ap_values else 0.0,
            "top_classes": top_curve_data,
        },
    }
    return curves, per_class


def save_per_class_csv(
    path: str,
    per_class_metrics: Dict[int, Dict[str, float]],
    curve_metrics: Dict[int, Dict[str, float]],
) -> None:
    _ensure_dir(os.path.dirname(path))
    fieldnames = [
        "class_id",
        "support",
        "predicted",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for class_id in sorted(per_class_metrics):
            writer.writerow({
                "class_id": class_id,
                **per_class_metrics[class_id],
                **curve_metrics.get(class_id, {"roc_auc": "", "average_precision": ""}),
            })


def plot_roc_curve(path: str, roc_data: Dict[str, Any]) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(roc_data["micro_fpr"], roc_data["micro_tpr"], label=f"micro AUC={roc_data['micro_auc']:.4f}", linewidth=2)
    plt.plot(roc_data["macro_fpr"], roc_data["macro_tpr"], label=f"macro AUC={roc_data['macro_auc']:.4f}", linewidth=2)
    for class_id, curve in roc_data["top_classes"].items():
        plt.plot(curve["roc_fpr"], curve["roc_tpr"], linestyle="--", alpha=0.75, label=f"class {class_id}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_pr_curve(path: str, pr_data: Dict[str, Any]) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(pr_data["micro_recall"], pr_data["micro_precision"], label=f"micro AP={pr_data['micro_ap']:.4f}", linewidth=2)
    plt.plot(pr_data["macro_recall"], pr_data["macro_precision"], label=f"macro AP={pr_data['macro_ap']:.4f}", linewidth=2)
    for class_id, curve in pr_data["top_classes"].items():
        plt.plot(curve["pr_recall"], curve["pr_precision"], linestyle="--", alpha=0.75, label=f"class {class_id}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("One-vs-Rest Precision-Recall Curves")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_confusion_matrix(path: str, confusion: np.ndarray, labels: np.ndarray) -> None:
    support = confusion.sum(axis=1)
    order = np.argsort(support)[::-1][: min(CONFUSION_TOP_N, len(labels))]
    matrix = confusion[np.ix_(order, order)].astype(np.float64)
    matrix = _safe_div(matrix, matrix.sum(axis=1, keepdims=True))

    plt.figure(figsize=(11, 9))
    plt.imshow(matrix, cmap="magma", aspect="auto", interpolation="nearest")
    tick_labels = labels[order]
    tick_positions = np.arange(len(order))
    plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=6)
    plt.yticks(tick_positions, tick_labels, fontsize=6)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Normalized confusion matrix (top {len(order)} classes by support)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_cluster_diagnostics(path: str, cluster_sizes: np.ndarray, cluster_purities: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sorted_sizes = np.sort(cluster_sizes)[::-1]
    axes[0].plot(np.arange(1, len(sorted_sizes) + 1), sorted_sizes, linewidth=2)
    axes[0].set_title("Cluster size rank curve")
    axes[0].set_xlabel("Cluster rank")
    axes[0].set_ylabel("Samples in cluster")

    axes[1].hist(cluster_purities, bins=30, color="#4c72b0", alpha=0.9)
    axes[1].set_title("Cluster purity distribution")
    axes[1].set_xlabel("Purity")
    axes[1].set_ylabel("Cluster count")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def compute_topk_accuracy(
    cluster_probs: np.ndarray,
    eval_assignments: np.ndarray,
    eval_labels: np.ndarray,
    k: int,
    chunk_size: int = 20_000,
) -> float:
    hits = 0
    total = int(eval_labels.shape[0])
    if total == 0:
        return 0.0
    k = max(1, min(k, cluster_probs.shape[1]))

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        batch_probs = cluster_probs[eval_assignments[start:end]]
        kth = batch_probs.shape[1] - k
        topk = np.argpartition(batch_probs, kth=kth, axis=1)[:, -k:]
        hits += int(np.any(topk == eval_labels[start:end, None], axis=1).sum())
    return float(hits / total)


def evaluate_saved_result(
    method_name: str,
    result_npz_path: str,
    feature_dir: str,
    output_dir: str,
    split: str = "val",
) -> EvaluationArtifacts:
    _ensure_dir(output_dir)

    data = np.load(result_npz_path)
    train_assignments = data["train_assignments"].astype(np.int64, copy=False)
    eval_assignments = data[f"{split}_assignments"].astype(np.int64, copy=False)
    train_labels = np.load(os.path.join(feature_dir, "train_labels.npy")).astype(np.int64, copy=False)
    eval_labels = np.load(os.path.join(feature_dir, f"{split}_labels.npy")).astype(np.int64, copy=False)

    n_classes = int(max(train_labels.max(), eval_labels.max())) + 1
    n_clusters = int(max(train_assignments.max(), eval_assignments.max())) + 1
    if np.any(train_assignments < 0) or np.any(eval_assignments < 0):
        raise ValueError("Cluster assignments contain negative indices")
    if np.any(train_assignments >= n_clusters) or np.any(eval_assignments >= n_clusters):
        raise ValueError("Cluster assignments contain out-of-range indices")

    class_labels = np.arange(n_classes, dtype=np.int64)

    cluster_label_counts, cluster_probs, majority_labels, cluster_purities = build_cluster_label_model(
        train_labels,
        train_assignments,
        n_classes,
        n_clusters,
    )
    cluster_probs = cluster_probs.astype(np.float32, copy=False)
    predicted_labels = majority_labels[eval_assignments]
    confusion = build_contingency(eval_labels, predicted_labels, n_classes, n_classes)

    clustering = compute_clustering_metrics(eval_labels, eval_assignments)
    classification_summary, per_class_metrics = compute_classification_metrics(confusion, class_labels)

    label_cluster_counts = build_contingency(eval_labels, eval_assignments, n_classes, n_clusters)
    curves, per_class_curve_metrics = compute_multiclass_curve_metrics(cluster_probs, label_cluster_counts)

    top1 = compute_topk_accuracy(cluster_probs, eval_assignments, eval_labels, k=1)
    top5 = compute_topk_accuracy(cluster_probs, eval_assignments, eval_labels, k=min(5, n_classes))
    classification_summary["top1_accuracy"] = top1
    classification_summary["top5_accuracy"] = top5

    summary = {
        "method": method_name,
        "split": split,
        "n_samples": int(eval_labels.shape[0]),
        "n_classes": int(n_classes),
        "n_clusters": int(n_clusters),
        "clustering": clustering,
        "classification": classification_summary,
        "classification_mapping": {
            "hard_prediction": "cluster->majority IN1K label from train split",
            "soft_score": "P(class|cluster) estimated from train split",
        },
        "curves": {
            "roc_auc_micro": curves["roc"]["micro_auc"],
            "roc_auc_macro": curves["roc"]["macro_auc"],
            "average_precision_micro": curves["pr"]["micro_ap"],
            "average_precision_macro": curves["pr"]["macro_ap"],
        },
        "cluster_stats": {
            "mean_cluster_size": float(cluster_label_counts.sum(axis=1).mean()),
            "std_cluster_size": float(cluster_label_counts.sum(axis=1).std()),
            "mean_cluster_purity": float(np.mean(cluster_purities)),
            "median_cluster_purity": float(np.median(cluster_purities)),
            "min_cluster_purity": float(np.min(cluster_purities)),
            "max_cluster_purity": float(np.max(cluster_purities)),
            "non_empty_clusters": int(np.count_nonzero(cluster_label_counts.sum(axis=1))),
        },
    }

    detailed_json_path = os.path.join(output_dir, "detailed_metrics.json")
    per_class_csv_path = os.path.join(output_dir, "per_class_metrics.csv")
    curve_data_path = os.path.join(output_dir, "curve_data.json")
    roc_plot_path = os.path.join(output_dir, "roc_curve.png")
    pr_plot_path = os.path.join(output_dir, "pr_curve.png")
    confusion_plot_path = os.path.join(output_dir, "confusion_matrix_top50.png")
    cluster_plot_path = os.path.join(output_dir, "cluster_diagnostics.png")

    with open(detailed_json_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(curve_data_path, "w") as handle:
        json.dump(curves, handle)
    save_per_class_csv(per_class_csv_path, per_class_metrics, per_class_curve_metrics)
    plot_roc_curve(roc_plot_path, curves["roc"])
    plot_pr_curve(pr_plot_path, curves["pr"])
    plot_confusion_matrix(confusion_plot_path, confusion, class_labels)
    plot_cluster_diagnostics(cluster_plot_path, cluster_label_counts.sum(axis=1), cluster_purities)

    return EvaluationArtifacts(
        summary=summary,
        paths={
            "detailed_json": detailed_json_path,
            "curve_json": curve_data_path,
            "per_class_csv": per_class_csv_path,
            "roc_plot": roc_plot_path,
            "pr_plot": pr_plot_path,
            "confusion_plot": confusion_plot_path,
            "cluster_plot": cluster_plot_path,
        },
    )
