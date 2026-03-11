"""
Benchmark: TorchFAISS Distributed KMeans on 20x ImageNet CLIP features.

Uses memory-mapped files to avoid loading 75GB on every rank.

Usage:
    torchrun --nproc_per_node=8 benchmark_20x.py
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
    parser.add_argument("--feature_dir", type=str, default=os.path.join(repo_dir, "features_20x"))
    parser.add_argument("--result_dir", type=str, default=os.path.join(repo_dir, "results_20x"))
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--niter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_triton", action="store_true")
    parser.add_argument("--int8_assign", action="store_true")
    parser.add_argument("--int8_fixed_scale", type=float, default=None)
    parser.add_argument("--max_points_per_centroid", type=int, default=256)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    # Memory-map the large training features to avoid 8x copies in RAM
    train_features = np.load(os.path.join(args.feature_dir, "train_features.npy"), mmap_mode="r")
    n_train, d = train_features.shape

    if rank == 0:
        print("=" * 80)
        print("TorchFAISS Distributed KMeans Benchmark (20x)")
        print("=" * 80)
        print(f"Train: {train_features.shape}, d={d}")
        print(f"K={args.k}, niter={args.niter}, world_size={world_size}")
        print("=" * 80)

    # Compute shard boundaries
    shard_size = n_train // world_size
    start = rank * shard_size
    end = start + shard_size if rank < world_size - 1 else n_train

    if rank == 0:
        print(f"Loading local shard [{start}:{end}] ({end - start} vectors) ...")

    # Load only this rank's shard into contiguous memory
    train_local = np.array(train_features[start:end])  # copy from mmap

    if rank == 0:
        print(f"Local shard loaded: {train_local.shape}")

    # ========================================================================
    # Distributed KMeans
    # ========================================================================
    dist.barrier()
    km = TorchKmeans(
        d=d, k=args.k, niter=args.niter,
        verbose=True,
        seed=args.seed,
        distributed=True,
        bf16=args.bf16,
        use_triton=args.use_triton,
        int8_assign=args.int8_assign,
        int8_fixed_scale=args.int8_fixed_scale,
        max_points_per_centroid=args.max_points_per_centroid,
    )

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(local_rank)
    km.train(train_local)
    train_time = time.time() - t0
    train_peak_mem_mb = torch.cuda.max_memory_allocated(local_rank) / (1024 ** 2)

    del train_local  # free memory before assign

    if rank == 0:
        # Assign full train set (streaming from mmap, GPU-batched)
        print(f"\nAssigning train set ({n_train} vectors) ...")
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats(local_rank)
        # Read from mmap in large chunks to reduce I/O overhead, but assign in GPU batches
        chunk_size = 1_000_000  # 1M vectors per read from disk
        D_train_list = []
        I_train_list = []
        for cs in range(0, n_train, chunk_size):
            ce = min(cs + chunk_size, n_train)
            chunk = np.array(train_features[cs:ce])
            D_chunk, I_chunk = km.assign(chunk)
            D_train_list.append(D_chunk)
            I_train_list.append(I_chunk)
            print(f"  Assigned {ce}/{n_train} ...", flush=True)
        D_train = np.concatenate(D_train_list)
        I_train = np.concatenate(I_train_list)
        assign_train_time = time.time() - t0
        assign_train_peak_mem_mb = torch.cuda.max_memory_allocated(local_rank) / (1024 ** 2)
        train_obj = D_train.sum()

        # Assign val set
        val_features = np.load(os.path.join(args.feature_dir, "val_features.npy"))
        val_labels = np.load(os.path.join(args.feature_dir, "val_labels.npy"))
        print(f"Assigning val set ({val_features.shape[0]} vectors) ...")
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats(local_rank)
        D_val, I_val = km.assign(val_features)
        assign_val_time = time.time() - t0
        assign_val_peak_mem_mb = torch.cuda.max_memory_allocated(local_rank) / (1024 ** 2)
        val_obj = D_val.sum()

        # Metrics — use labels (tiled labels are periodic)
        train_labels = np.load(os.path.join(args.feature_dir, "train_labels.npy"))
        nmi_train, purity_train = compute_metrics(train_labels, I_train)
        nmi_val, purity_val = compute_metrics(val_labels, I_val)

        print("\n" + "=" * 80)
        print(f"RESULTS: TorchFAISS Distributed ({world_size} GPUs) — 20x Data")
        print("=" * 80)
        print(f"  Train vectors:  {n_train:,}")
        print(f"  Val vectors:    {val_features.shape[0]:,}")
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
            "int8_fixed_scale": args.int8_fixed_scale,
            "max_points_per_centroid": args.max_points_per_centroid,
            "n_train": int(n_train),
            "n_val": int(val_features.shape[0]),
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
