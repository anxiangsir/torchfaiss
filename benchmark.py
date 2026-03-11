"""
Benchmark: TorchFAISS Distributed KMeans on ImageNet CLIP features.

Usage:
    torchrun --nproc_per_node=8 benchmark.py

Saves results to results/torchfaiss_result.npz for comparison.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchfaiss import TorchKmeans


def compute_metrics(labels_true, labels_pred):
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
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_triton", action="store_true")
    parser.add_argument("--int8_assign", action="store_true")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    # Load features
    train_features = np.load(os.path.join(args.feature_dir, "train_features.npy"))
    train_labels = np.load(os.path.join(args.feature_dir, "train_labels.npy"))
    val_features = np.load(os.path.join(args.feature_dir, "val_features.npy"))
    val_labels = np.load(os.path.join(args.feature_dir, "val_labels.npy"))

    if rank == 0:
        print("=" * 80)
        print("TorchFAISS Distributed KMeans Benchmark")
        print("=" * 80)
        print(f"Train: {train_features.shape}, Val: {val_features.shape}")
        print(f"K={args.k}, niter={args.niter}, world_size={world_size}")
        print("=" * 80)

    # Shard training data
    n = train_features.shape[0]
    shard_size = n // world_size
    start = rank * shard_size
    end = start + shard_size if rank < world_size - 1 else n
    train_local = train_features[start:end]

    if rank == 0:
        print(f"Local shard: {train_local.shape}")

    # ========================================================================
    # Distributed KMeans (all ranks participate)
    # ========================================================================
    dist.barrier()
    km = TorchKmeans(
        d=train_features.shape[1], k=args.k, niter=args.niter,
        verbose=True,
        seed=args.seed,
        distributed=True,
        bf16=args.bf16,
        use_triton=args.use_triton,
        int8_assign=args.int8_assign,
    )

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(local_rank)
    km.train(train_local)
    train_time = time.time() - t0
    train_peak_mem_mb = torch.cuda.max_memory_allocated(local_rank) / (1024 ** 2)

    if rank == 0:
        # Assign full train set
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats(local_rank)
        D_train, I_train = km.assign(train_features)
        assign_train_time = time.time() - t0
        assign_train_peak_mem_mb = torch.cuda.max_memory_allocated(local_rank) / (1024 ** 2)
        train_obj = D_train.sum()

        # Assign val set
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats(local_rank)
        D_val, I_val = km.assign(val_features)
        assign_val_time = time.time() - t0
        assign_val_peak_mem_mb = torch.cuda.max_memory_allocated(local_rank) / (1024 ** 2)
        val_obj = D_val.sum()

        # Metrics
        nmi_train, purity_train = compute_metrics(train_labels, I_train)
        nmi_val, purity_val = compute_metrics(val_labels, I_val)

        print("\n" + "=" * 80)
        print(f"RESULTS: TorchFAISS Distributed ({world_size} GPUs)")
        print("=" * 80)
        print(f"  Train time:     {train_time:.2f} s")
        print(f"  Assign time:    {assign_train_time:.2f} s (train), {assign_val_time:.2f} s (val)")
        print(f"  Peak mem:       {train_peak_mem_mb:.1f} MB (train), {assign_train_peak_mem_mb:.1f} MB (assign-train), {assign_val_peak_mem_mb:.1f} MB (assign-val)")
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
            os.path.join(args.result_dir, "torchfaiss_result.npz"),
            centroids=centroids,
            train_assignments=I_train,
            val_assignments=I_val,
            train_distances=D_train,
            val_distances=D_val,
        )
        result_json = {
            "method": f"TorchFAISS Dist({world_size}GPU)",
            "k": args.k, "niter": args.niter,
            "bf16": args.bf16,
            "use_triton": args.use_triton,
            "int8_assign": args.int8_assign,
            "train_time": round(train_time, 3),
            "assign_train_time": round(assign_train_time, 3),
            "assign_val_time": round(assign_val_time, 3),
            "train_peak_mem_mb": round(train_peak_mem_mb, 3),
            "assign_train_peak_mem_mb": round(assign_train_peak_mem_mb, 3),
            "assign_val_peak_mem_mb": round(assign_val_peak_mem_mb, 3),
            "train_obj": float(train_obj),
            "val_obj": float(val_obj),
            "train_nmi": round(nmi_train, 4),
            "train_purity": round(purity_train, 4),
            "val_nmi": round(nmi_val, 4),
            "val_purity": round(purity_val, 4),
        }
        with open(os.path.join(args.result_dir, "torchfaiss_result.json"), "w") as f:
            json.dump(result_json, f, indent=2)
        print(f"\nResults saved to {args.result_dir}/")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
