import argparse
import json
import os
import time
from collections import Counter
from typing import Dict, Any, List

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score

from torchfaiss import TorchKmeans


def run_mode(
    mode_name: str,
    train_features: np.ndarray,
    val_features: np.ndarray,
    k: int,
    niter: int,
    seed: int,
    batch_size: int,
) -> Dict[str, Any]:
    use_bf16 = mode_name == "bf16"
    use_triton = mode_name in {"triton", "triton_int8"}
    use_int8 = mode_name == "triton_int8"

    km = TorchKmeans(
        d=train_features.shape[1],
        k=k,
        niter=niter,
        seed=seed,
        verbose=False,
        gpu=True,
        distributed=False,
        bf16=use_bf16,
        use_triton=use_triton,
        int8_assign=use_int8,
    )

    if km._compiled_pairwise is not None:
        with torch.no_grad():
            dummy_centroids = torch.randn(k, train_features.shape[1], device=km.device, dtype=torch.float32)
            dummy_c_sq = (dummy_centroids * dummy_centroids).sum(dim=1)
            dummy_x_train = torch.randn(min(131072, train_features.shape[0]), train_features.shape[1], device=km.device, dtype=torch.float32)
            dummy_x_assign = torch.randn(min(batch_size, val_features.shape[0]), train_features.shape[1], device=km.device, dtype=torch.float32)
            _ = km._compiled_pairwise(dummy_x_train, dummy_centroids, dummy_c_sq)
            _ = km._compiled_pairwise(dummy_x_assign, dummy_centroids, dummy_c_sq)

    device_idx = km.device.index if km.device.index is not None else torch.cuda.current_device()

    torch.cuda.synchronize(device_idx)
    torch.cuda.reset_peak_memory_stats(device_idx)
    t0 = time.time()
    km.train(train_features)
    torch.cuda.synchronize(device_idx)
    train_time = time.time() - t0
    train_peak_mem_mb = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2)

    torch.cuda.synchronize(device_idx)
    torch.cuda.reset_peak_memory_stats(device_idx)
    t0 = time.time()
    d_val, i_val = km.assign(val_features, batch_size=batch_size)
    torch.cuda.synchronize(device_idx)
    assign_time = time.time() - t0
    assign_peak_mem_mb = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2)

    return {
        "mode": mode_name,
        "train_time": float(train_time),
        "assign_time": float(assign_time),
        "val_obj": float(d_val.sum()),
        "train_peak_mem_mb": float(train_peak_mem_mb),
        "assign_peak_mem_mb": float(assign_peak_mem_mb),
        "val_assignments": i_val,
        "val_distances": d_val,
    }


def compute_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    total = len(labels_true)
    purity_sum = 0
    for cid in np.unique(labels_pred):
        mask = labels_pred == cid
        if mask.sum() == 0:
            continue
        counter = Counter(labels_true[mask])
        purity_sum += counter.most_common(1)[0][1]
    return float(purity_sum / total)


def main() -> None:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default=os.path.join(repo_dir, "features"))
    parser.add_argument("--output", type=str, default=os.path.join(repo_dir, "results", "precision_modes.json"))
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--niter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=65536)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_precision_modes.py requires CUDA")

    train_features = np.load(os.path.join(args.feature_dir, "train_features.npy"))
    val_features = np.load(os.path.join(args.feature_dir, "val_features.npy"))
    val_labels = np.load(os.path.join(args.feature_dir, "val_labels.npy"))

    modes: List[str] = ["fp32", "bf16", "triton", "triton_int8"]
    raw_results: List[Dict[str, Any]] = []

    warm_km = TorchKmeans(d=train_features.shape[1], k=min(args.k, 128), niter=1, seed=args.seed, verbose=False, gpu=True)
    warm_km.train(train_features[: min(65536, train_features.shape[0])])
    _ = warm_km.assign(val_features[: min(32768, val_features.shape[0])], batch_size=min(args.batch_size, 32768))

    for mode in modes:
        print(f"Running mode: {mode}", flush=True)
        raw_results.append(
            run_mode(
                mode_name=mode,
                train_features=train_features,
                val_features=val_features,
                k=args.k,
                niter=args.niter,
                seed=args.seed,
                batch_size=args.batch_size,
            )
        )

    baseline = next(item for item in raw_results if item["mode"] == "fp32")
    baseline_assign = baseline["val_assignments"]
    baseline_obj = baseline["val_obj"]

    summary: Dict[str, Any] = {"config": vars(args), "modes": {}}
    for item in raw_results:
        agree = float((item["val_assignments"] == baseline_assign).mean())
        assign_nmi = float(normalized_mutual_info_score(baseline_assign, item["val_assignments"]))
        val_nmi = float(normalized_mutual_info_score(val_labels, item["val_assignments"]))
        val_purity = compute_purity(val_labels, item["val_assignments"])
        obj_diff_pct = float(abs(item["val_obj"] - baseline_obj) / max(abs(baseline_obj), 1e-12) * 100.0)
        summary["modes"][item["mode"]] = {
            "train_time": round(item["train_time"], 4),
            "assign_time": round(item["assign_time"], 4),
            "speedup_vs_fp32_train": round(baseline["train_time"] / max(item["train_time"], 1e-12), 4),
            "speedup_vs_fp32_assign": round(baseline["assign_time"] / max(item["assign_time"], 1e-12), 4),
            "val_obj": round(item["val_obj"], 4),
            "val_obj_diff_pct_vs_fp32": round(obj_diff_pct, 6),
            "val_assignment_agreement_vs_fp32": round(agree, 6),
            "val_assignment_nmi_vs_fp32": round(assign_nmi, 6),
            "val_nmi_vs_labels": round(val_nmi, 6),
            "val_purity_vs_labels": round(val_purity, 6),
            "train_peak_mem_mb": round(item["train_peak_mem_mb"], 3),
            "assign_peak_mem_mb": round(item["assign_peak_mem_mb"], 3),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
