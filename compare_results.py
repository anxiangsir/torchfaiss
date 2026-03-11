import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from eval_utils import evaluate_saved_result, slugify


def infer_feature_dir(result_dir: str) -> str:
    normalized = os.path.abspath(result_dir)
    if normalized.endswith("results_20x"):
        return os.path.join(os.path.dirname(normalized), "features_20x")
    return os.path.join(os.path.dirname(normalized), "features")


def discover_results(result_dir: str) -> List[Tuple[str, str, str]]:
    items = []
    for json_name, npz_name in [
        ("torchfaiss_result.json", "torchfaiss_result.npz"),
        ("faiss_result.json", "faiss_result.npz"),
    ]:
        json_path = os.path.join(result_dir, json_name)
        npz_path = os.path.join(result_dir, npz_name)
        if not os.path.exists(json_path) or not os.path.exists(npz_path):
            continue
        with open(json_path) as handle:
            payload = json.load(handle)
        items.append((payload["method"], json_path, npz_path))
    return items


def plot_speed_comparison(report_root: str, runtime_results: List[Dict[str, Any]]) -> str:
    methods = [str(item["method"]) for item in runtime_results]
    train_times = [float(item.get("train_time", 0.0)) for item in runtime_results]
    assign_times = [
        float(item.get("assign_train_time", 0.0)) + float(item.get("assign_val_time", 0.0))
        for item in runtime_results
    ]

    x = np.arange(len(methods), dtype=np.float64)
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_train = ax.bar(x - width / 2, train_times, width, label="Train time (s)", color="#4c72b0")
    bars_assign = ax.bar(x + width / 2, assign_times, width, label="Assign time (s)", color="#dd8452")

    for bars in (bars_train, bars_assign):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(max(train_times + assign_times) * 0.01, 0.02),
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=8)
    ax.set_ylabel("Seconds")
    ax.set_title("FAISS vs TorchFAISS Speed Comparison")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()

    out_path = os.path.join(report_root, "speed_comparison.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_curve_comparison(
    report_root: str,
    report_manifest: Dict[str, Dict[str, Any]],
    result_dir: str,
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2))

    color_cycle = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    roc_micro_fprs: List[np.ndarray] = []
    roc_micro_tprs: List[np.ndarray] = []
    pr_micro_recalls: List[np.ndarray] = []
    pr_micro_precisions: List[np.ndarray] = []
    for idx, (method_name, payload) in enumerate(report_manifest.items()):
        curve_rel = payload["artifacts"]["curve_json"]
        curve_path = os.path.join(result_dir, curve_rel)
        with open(curve_path) as handle:
            curve_data = json.load(handle)

        roc = curve_data["roc"]
        pr = curve_data["pr"]
        color = color_cycle[idx % len(color_cycle)]

        roc_fpr = np.asarray(roc["micro_fpr"], dtype=np.float64)
        roc_tpr = np.asarray(roc["micro_tpr"], dtype=np.float64)
        pr_recall = np.asarray(pr["micro_recall"], dtype=np.float64)
        pr_precision = np.asarray(pr["micro_precision"], dtype=np.float64)
        roc_micro_fprs.append(roc_fpr)
        roc_micro_tprs.append(roc_tpr)
        pr_micro_recalls.append(pr_recall)
        pr_micro_precisions.append(pr_precision)

        axes[0, 0].plot(
            roc["micro_fpr"],
            roc["micro_tpr"],
            color=color,
            linewidth=2,
            label=f"{method_name} (AUC={roc['micro_auc']:.4f})",
        )
        axes[0, 1].plot(
            pr["micro_recall"],
            pr["micro_precision"],
            color=color,
            linewidth=2,
            label=f"{method_name} (AP={pr['micro_ap']:.4f})",
        )

    axes[0, 0].plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
    axes[0, 0].set_title("Micro ROC (full range)")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].grid(alpha=0.25, linestyle=":")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title("Micro PR (full range)")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].grid(alpha=0.25, linestyle=":")
    axes[0, 1].legend(fontsize=8)

    for idx, method_name in enumerate(report_manifest.keys()):
        color = color_cycle[idx % len(color_cycle)]
        axes[1, 0].plot(
            roc_micro_fprs[idx],
            roc_micro_tprs[idx],
            color=color,
            linewidth=2,
            label=method_name,
        )
        axes[1, 1].plot(
            pr_micro_recalls[idx],
            pr_micro_precisions[idx],
            color=color,
            linewidth=2,
            label=method_name,
        )

    axes[1, 0].set_title("Micro ROC (zoomed: FPR 0~0.10)")
    axes[1, 0].set_xlabel("False Positive Rate")
    axes[1, 0].set_ylabel("True Positive Rate")
    axes[1, 0].set_xlim(0.0, 0.10)
    axes[1, 0].set_ylim(0.85, 1.0)
    axes[1, 0].grid(alpha=0.3, linestyle=":")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title("Micro PR (zoomed: Recall 0.80~1.00)")
    axes[1, 1].set_xlabel("Recall")
    axes[1, 1].set_ylabel("Precision")
    axes[1, 1].set_xlim(0.80, 1.0)
    axes[1, 1].set_ylim(0.0, 0.7)
    axes[1, 1].grid(alpha=0.3, linestyle=":")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("FAISS vs TorchFAISS Curve Comparison (with zoomed views)")
    fig.tight_layout()
    out_path = os.path.join(report_root, "curve_comparison.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def print_runtime_table(runtime_results: List[Dict[str, Any]]) -> None:
    print(f"\n{'=' * 132}")
    print("  KMeans Benchmark Comparison: TorchFAISS vs FAISS")
    print(f"{'=' * 132}")
    header = (
        f"{'Method':<28} {'Mode':<14} {'Train(s)':<10} {'Assign(s)':<10} {'Mem(MB)':<10} {'ValObj':<14} "
        f"{'NMI':<8} {'Purity':<8} {'Top1':<8} {'Top5':<8} {'P':<8} {'R':<8} {'F1':<8} {'ROC-AUC':<9} {'PR-AUC':<9}"
    )
    print(header)
    print("-" * len(header))

    for item in runtime_results:
        assign_time = item.get("assign_train_time", 0.0) + item.get("assign_val_time", 0.0)
        mode_parts: List[str] = ["fp32"]
        if item.get("bf16", False):
            mode_parts.append("bf16")
        if item.get("use_triton", False):
            mode_parts.append("triton")
        if item.get("int8_assign", False):
            mode_parts.append("int8")
        mode = "+".join(mode_parts)
        peak_mem = max(
            float(item.get("train_peak_mem_mb", 0.0)),
            float(item.get("assign_train_peak_mem_mb", 0.0)),
            float(item.get("assign_val_peak_mem_mb", 0.0)),
        )
        eval_summary = item.get("evaluation", {})
        clustering = eval_summary.get("clustering", {})
        classification = eval_summary.get("classification", {})
        curves = eval_summary.get("curves", {})
        print(
            f"{item['method']:<28} "
            f"{mode:<14} "
            f"{item.get('train_time', 0.0):<10.2f} "
            f"{assign_time:<10.2f} "
            f"{peak_mem:<10.1f} "
            f"{item.get('val_obj', 0.0):<14.0f} "
            f"{clustering.get('nmi', 0.0):<8.4f} "
            f"{clustering.get('purity', 0.0):<8.4f} "
            f"{classification.get('top1_accuracy', 0.0):<8.4f} "
            f"{classification.get('top5_accuracy', 0.0):<8.4f} "
            f"{classification.get('precision_macro', 0.0):<8.4f} "
            f"{classification.get('recall_macro', 0.0):<8.4f} "
            f"{classification.get('f1_macro', 0.0):<8.4f} "
            f"{curves.get('roc_auc_macro', 0.0):<9.4f} "
            f"{curves.get('average_precision_macro', 0.0):<9.4f}"
        )


def summarize_cross_method(result_dir: str, runtime_results: List[Dict[str, Any]]) -> Dict[str, float]:
    tf_path = os.path.join(result_dir, "torchfaiss_result.npz")
    fa_path = os.path.join(result_dir, "faiss_result.npz")
    if not (os.path.exists(tf_path) and os.path.exists(fa_path)):
        return {}

    tf = np.load(tf_path)
    fa = np.load(fa_path)

    train_agree = float((tf["train_assignments"] == fa["train_assignments"]).mean())
    val_agree = float((tf["val_assignments"] == fa["val_assignments"]).mean())
    cost = cdist(tf["centroids"], fa["centroids"], metric="sqeuclidean")
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_dist = cost[row_ind, col_ind]

    remap = np.empty(cost.shape[0], dtype=np.int64)
    remap[row_ind] = col_ind
    remapped_train_agree = float((remap[tf["train_assignments"]] == fa["train_assignments"]).mean())
    remapped_val_agree = float((remap[tf["val_assignments"]] == fa["val_assignments"]).mean())

    tf_json = next(item for item in runtime_results if "TorchFAISS" in item["method"])
    fa_json = next(item for item in runtime_results if "FAISS" in item["method"] and "TorchFAISS" not in item["method"])
    val_obj_diff_pct = float(abs(tf_json["val_obj"] - fa_json["val_obj"]) / max(fa_json["val_obj"], 1e-12) * 100.0)
    val_nmi_diff = float(
        abs(tf_json["evaluation"]["clustering"]["nmi"] - fa_json["evaluation"]["clustering"]["nmi"])
    )
    speedup = float(fa_json["train_time"] / tf_json["train_time"]) if tf_json["train_time"] else 0.0

    summary = {
        "train_assignment_agreement": train_agree,
        "val_assignment_agreement": val_agree,
        "centroid_l2_mean": float(matched_dist.mean()),
        "centroid_l2_max": float(matched_dist.max()),
        "centroid_l2_min": float(matched_dist.min()),
        "remapped_train_assignment_agreement": remapped_train_agree,
        "remapped_val_assignment_agreement": remapped_val_agree,
        "val_objective_diff_pct": val_obj_diff_pct,
        "val_nmi_diff": val_nmi_diff,
        "training_speedup": speedup,
    }

    print(f"\n{'=' * 132}")
    print("  Cross-Method Analysis")
    print(f"{'=' * 132}")
    print(f"  Train assignment agreement:             {train_agree:.4f} ({train_agree * 100:.1f}%)")
    print(f"  Val assignment agreement:               {val_agree:.4f} ({val_agree * 100:.1f}%)")
    print(f"  Centroid L2 distance (Hungarian mean):  {matched_dist.mean():.6f}")
    print(f"  Centroid L2 distance (Hungarian max):   {matched_dist.max():.6f}")
    print(f"  Centroid L2 distance (Hungarian min):   {matched_dist.min():.8f}")
    print(f"  Remapped train assignment agreement:    {remapped_train_agree:.4f} ({remapped_train_agree * 100:.1f}%)")
    print(f"  Remapped val assignment agreement:      {remapped_val_agree:.4f} ({remapped_val_agree * 100:.1f}%)")
    print(f"  Val objective difference:               {val_obj_diff_pct:.2f}%")
    print(f"  Val NMI difference:                     {val_nmi_diff:.4f}")
    print(f"  Training speedup (TorchFAISS vs FAISS): {speedup:.2f}x")
    return summary


def main() -> None:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default=os.path.join(repo_dir, "results"))
    parser.add_argument("--feature_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--report_subdir", type=str, default="evaluation_reports")
    args = parser.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    feature_dir = os.path.abspath(args.feature_dir) if args.feature_dir else infer_feature_dir(result_dir)
    report_root = os.path.join(result_dir, args.report_subdir)
    os.makedirs(report_root, exist_ok=True)

    discovered = discover_results(result_dir)
    if not discovered:
        print(f"No benchmark result pairs found in {result_dir}")
        return

    runtime_results: List[Dict[str, Any]] = []
    report_manifest: Dict[str, Dict[str, Any]] = {}
    for method_name, json_path, npz_path in discovered:
        with open(json_path) as handle:
            runtime_payload = json.load(handle)
        method_slug = slugify(method_name)
        output_dir = os.path.join(report_root, method_slug)
        artifacts = evaluate_saved_result(
            method_name=method_name,
            result_npz_path=npz_path,
            feature_dir=feature_dir,
            output_dir=output_dir,
            split=args.split,
        )
        runtime_payload["evaluation"] = artifacts.summary
        runtime_results.append(runtime_payload)
        report_manifest[method_name] = {
            "summary": artifacts.summary,
            "artifacts": {k: os.path.relpath(v, result_dir) for k, v in artifacts.paths.items()},
        }

    print_runtime_table(runtime_results)
    cross_method = summarize_cross_method(result_dir, runtime_results)
    speed_plot_path = plot_speed_comparison(report_root, runtime_results)
    curve_plot_path = plot_curve_comparison(report_root, report_manifest, result_dir)
    manifest_path = os.path.join(report_root, "summary.json")
    with open(manifest_path, "w") as handle:
        json.dump(
            {
                "result_dir": ".",
                "feature_dir": os.path.relpath(feature_dir, result_dir),
                "split": args.split,
                "methods": report_manifest,
                "cross_method": cross_method,
                "speed_plot": os.path.relpath(speed_plot_path, result_dir),
                "curve_comparison_plot": os.path.relpath(curve_plot_path, result_dir),
            },
            handle,
            indent=2,
        )

    print(f"\nDetailed evaluation artifacts saved under: {report_root}")
    print(f"Summary manifest: {manifest_path}")


if __name__ == "__main__":
    main()
