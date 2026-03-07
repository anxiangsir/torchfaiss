"""
Benchmark: FAISS KMeans (multi-GPU) on ImageNet CLIP features.

Runs as a single process with FAISS multi-GPU support (gpu=N).

Usage:
    python benchmark_faiss.py --ngpu 8

Saves results to results/faiss_result.npz for comparison.
"""

import os
import time
import json
import argparse
import numpy as np


def compute_metrics(labels_true, labels_pred):
    """Compute NMI and Purity."""
    from sklearn.metrics import normalized_mutual_info_score
    from collections import Counter
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    total = len(labels_true)
    purity_sum = 0
    for cid in np.unique(labels_pred):
        mask = labels_pred == cid
        if mask.sum() == 0:
            continue
        counter = Counter(labels_true[mask])
        purity_sum += counter.most_common(1)[0][1]
    purity = purity_sum / total
    return nmi, purity


def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default=os.path.join(repo_dir, "features"))
    parser.add_argument("--result_dir", type=str, default=os.path.join(repo_dir, "results"))
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--niter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ngpu", type=int, default=8,
                        help="Number of GPUs to use (default: 8, same as TorchFAISS for fair comparison)")
    args = parser.parse_args()

    import faiss
    print(f"FAISS version: {faiss.__version__}")
    print(f"FAISS GPU count: {faiss.get_num_gpus()}")
    ngpu = min(args.ngpu, faiss.get_num_gpus())
    print(f"Using {ngpu} GPU(s)")

    # Load features
    train_features = np.load(os.path.join(args.feature_dir, "train_features.npy"))
    train_labels = np.load(os.path.join(args.feature_dir, "train_labels.npy"))
    val_features = np.load(os.path.join(args.feature_dir, "val_features.npy"))
    val_labels = np.load(os.path.join(args.feature_dir, "val_labels.npy"))

    print("=" * 80)
    print("FAISS KMeans Benchmark")
    print("=" * 80)
    print(f"Train: {train_features.shape}, Val: {val_features.shape}")
    print(f"K={args.k}, niter={args.niter}, ngpu={ngpu}")
    print("=" * 80)

    d = train_features.shape[1]
    km = faiss.Kmeans(
        d, args.k,
        niter=args.niter,
        verbose=True,
        gpu=ngpu,
        seed=args.seed,
    )

    # Train
    t0 = time.time()
    km.train(train_features)
    train_time = time.time() - t0

    # Assign train set
    t0 = time.time()
    D_train, I_train = km.assign(train_features)
    assign_train_time = time.time() - t0
    train_obj = D_train.sum()

    # Assign val set using centroids
    t0 = time.time()
    D_val, I_val = km.assign(val_features)
    assign_val_time = time.time() - t0
    val_obj = D_val.sum()

    # Metrics
    nmi_train, purity_train = compute_metrics(train_labels, I_train)
    nmi_val, purity_val = compute_metrics(val_labels, I_val)

    print("\n" + "=" * 80)
    print(f"RESULTS: FAISS KMeans ({ngpu} GPU)")
    print("=" * 80)
    print(f"  Train time:     {train_time:.2f} s")
    print(f"  Assign time:    {assign_train_time:.2f} s (train), {assign_val_time:.2f} s (val)")
    print(f"  Train Obj:      {train_obj:.0f}")
    print(f"  Val Obj:        {val_obj:.0f}")
    print(f"  Train NMI:      {nmi_train:.4f}")
    print(f"  Train Purity:   {purity_train:.4f}")
    print(f"  Val NMI:        {nmi_val:.4f}")
    print(f"  Val Purity:     {purity_val:.4f}")

    # Save results
    centroids = km.centroids
    assert centroids is not None
    os.makedirs(args.result_dir, exist_ok=True)
    np.savez(
        os.path.join(args.result_dir, "faiss_result.npz"),
        centroids=centroids,
        train_assignments=I_train,
        val_assignments=I_val,
        train_distances=D_train,
        val_distances=D_val,
    )
    result_json = {
        "method": f"FAISS ({ngpu}GPU)",
        "k": args.k, "niter": args.niter,
        "ngpu": ngpu,
        "train_time": round(train_time, 3),
        "assign_train_time": round(assign_train_time, 3),
        "assign_val_time": round(assign_val_time, 3),
        "train_obj": float(train_obj),
        "val_obj": float(val_obj),
        "train_nmi": round(nmi_train, 4),
        "train_purity": round(purity_train, 4),
        "val_nmi": round(nmi_val, 4),
        "val_purity": round(purity_val, 4),
    }
    with open(os.path.join(args.result_dir, "faiss_result.json"), "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"\nResults saved to {args.result_dir}/")


if __name__ == "__main__":
    main()
