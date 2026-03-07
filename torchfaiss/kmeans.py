"""
TorchFAISS KMeans — pure PyTorch, distributed-ready.

Mirrors faiss.Kmeans API:
    km = TorchKmeans(d=768, k=1000, niter=20, gpu=True, verbose=True)
    km.train(x)           # x: np.ndarray or torch.Tensor [N, d]
    D, I = km.assign(x)   # D: distances [N], I: cluster ids [N]
    centroids = km.centroids  # np.ndarray [k, d]

Distributed usage:
    km = TorchKmeans(d=768, k=1000, niter=20, distributed=True, verbose=True)
    km.train(x_local)     # x_local: this rank's shard of data
    D, I = km.assign(x)   # works on full data (local operation)
"""

import time
import numpy as np
import torch
import torch.distributed as dist
from typing import Optional, Tuple, Union

ArrayLike = Union[np.ndarray, torch.Tensor]


class TorchKmeans:
    """
    Drop-in replacement for faiss.Kmeans, implemented in pure PyTorch.
    
    Supports distributed training across multiple GPUs via torch.distributed.
    The algorithm follows FAISS exactly:
      1. Random initialization (or kmeans++ if specified)
      2. Lloyd's iterations: assign → update centroids
      3. Handle empty clusters by splitting largest cluster
      4. Multiple restarts (nredo) and keep best by objective
    
    Parameters
    ----------
    d : int
        Dimension of vectors.
    k : int
        Number of centroids.
    niter : int
        Number of Lloyd iterations per restart.
    nredo : int
        Number of independent restarts (keeps best).
    verbose : bool
        Print progress info.
    spherical : bool
        If True, L2-normalize centroids after each update (for cosine similarity).
    seed : int
        Random seed for reproducibility.
    max_points_per_centroid : int
        Subsample training data if N > k * max_points_per_centroid.
    min_points_per_centroid : int
        Warn if N < k * min_points_per_centroid.
    frozen_centroids : bool
        If True, do not update centroids (just run assignment).
    gpu : bool or int
        True = auto-select GPU, int = specific GPU, False = CPU.
    distributed : bool
        If True, use torch.distributed for multi-GPU/multi-node.
        Requires dist.init_process_group() to be called beforehand.
    """

    def __init__(
        self,
        d: int,
        k: int,
        niter: int = 20,
        nredo: int = 1,
        verbose: bool = False,
        spherical: bool = False,
        seed: int = 1234,
        max_points_per_centroid: int = 256,
        min_points_per_centroid: int = 39,
        frozen_centroids: bool = False,
        gpu: Union[bool, int] = False,
        distributed: bool = False,
        bf16: bool = False,
    ):
        self.d = d
        self.k = k
        self.niter = niter
        self.nredo = nredo
        self.verbose = verbose
        self.spherical = spherical
        self.seed = seed
        self.max_points_per_centroid = max_points_per_centroid
        self.min_points_per_centroid = min_points_per_centroid
        self.frozen_centroids = frozen_centroids
        self.distributed = distributed
        self.bf16 = bf16
        self.centroids: Optional[np.ndarray] = None

        # Resolve device
        if distributed:
            # In distributed mode, each rank uses its own GPU
            if dist.is_initialized():
                local_rank = dist.get_rank() % torch.cuda.device_count()
                self.device = torch.device(f"cuda:{local_rank}")
            else:
                raise RuntimeError("distributed=True but torch.distributed is not initialized")
        elif gpu is True:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(gpu, int):
            self.device = torch.device(f"cuda:{gpu}")
        else:
            self.device = torch.device("cpu")

        self._bf16_enabled = (
            self.bf16
            and self.device.type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )

    def _to_tensor(self, x: ArrayLike) -> torch.Tensor:
        """Convert input to float32 torch.Tensor on self.device."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.float().to(self.device)

    def _dot(self, xb: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        if self._bf16_enabled:
            return (xb.to(torch.bfloat16) @ centroids.to(torch.bfloat16).t()).float()
        return xb @ centroids.t()

    def _log(self, msg: str):
        if self.verbose:
            if self.distributed:
                if dist.get_rank() == 0:
                    print(msg, flush=True)
            else:
                print(msg, flush=True)

    @property
    def _is_root(self) -> bool:
        if self.distributed:
            return dist.get_rank() == 0
        return True

    @property
    def _world_size(self) -> int:
        if self.distributed:
            return dist.get_world_size()
        return 1

    def _subsample(self, x: torch.Tensor) -> torch.Tensor:
        """Subsample data if too many points per centroid (matches FAISS behavior)."""
        n = x.shape[0]
        if self.distributed:
            # Get total N across all ranks
            total_n = torch.tensor([n], device=self.device, dtype=torch.long)
            dist.all_reduce(total_n, op=dist.ReduceOp.SUM)
            total_n = total_n.item()
        else:
            total_n = n

        max_points = self.k * self.max_points_per_centroid
        if total_n > max_points:
            # Each rank keeps proportional share
            local_max = int(max_points * (n / total_n))
            local_max = max(local_max, 1)
            if local_max < n:
                perm = torch.randperm(n, device=self.device)[:local_max]
                x = x[perm]
                self._log(f"  Subsampled from {total_n} to ~{max_points} points "
                         f"(local: {n} -> {local_max})")
        
        if total_n < self.k * self.min_points_per_centroid:
            self._log(f"  WARNING: only {total_n} training points for {self.k} centroids, "
                     f"recommended at least {self.k * self.min_points_per_centroid}")
        
        return x

    def _init_centroids(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        """
        Initialize centroids by random selection from data.
        In distributed mode, rank 0 picks and broadcasts.
        """
        g = torch.Generator(device='cpu')
        g.manual_seed(seed)

        if self.distributed:
            # Gather a subset of data to rank 0 for initialization
            # Each rank sends up to k samples
            n_local = min(x.shape[0], self.k)
            local_indices = torch.randperm(x.shape[0], generator=g)[:n_local]
            local_candidates = x[local_indices]  # [n_local, d]

            # Gather sizes
            n_tensor = torch.tensor([n_local], device=self.device, dtype=torch.long)
            all_n = [torch.zeros(1, dtype=torch.long, device=self.device) for _ in range(self._world_size)]
            dist.all_gather(all_n, n_tensor)
            all_n_list = [int(s.item()) for s in all_n]
            max_n = int(max(all_n_list))

            # Pad and gather candidates
            padded = torch.zeros(max_n, self.d, device=self.device)
            padded[:n_local] = local_candidates
            gathered = [torch.zeros(max_n, self.d, device=self.device) for _ in range(self._world_size)]
            dist.all_gather(gathered, padded)

            if self._is_root:
                # Concat all candidates, pick k
                candidates = torch.cat([gathered[i][:all_n_list[i]] for i in range(self._world_size)], dim=0)
                perm = torch.randperm(candidates.shape[0], generator=g)[:self.k]
                centroids = candidates[perm].contiguous()
            else:
                centroids = torch.zeros(self.k, self.d, device=self.device)

            dist.broadcast(centroids, src=0)
        else:
            perm = torch.randperm(x.shape[0], generator=g)[:self.k]
            centroids = x[perm].clone()

        if self.spherical:
            centroids = centroids / centroids.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        return centroids

    def _assign_batch(
        self, x: torch.Tensor, centroids: torch.Tensor, batch_size: int = 65536
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign each point to nearest centroid using batched matrix ops.
        
        Uses the identity: ||x - c||^2 = ||x||^2 - 2*x·c + ||c||^2
        
        Returns (D, I) where D[i] = distance to nearest centroid, I[i] = centroid index.
        """
        n = x.shape[0]
        D_all = torch.empty(n, device=self.device, dtype=torch.float32)
        I_all = torch.empty(n, device=self.device, dtype=torch.long)

        # Precompute ||c||^2  [k]
        c_sq = (centroids * centroids).sum(dim=1)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = x[start:end]  # [bs, d]

            # ||x||^2  [bs]
            x_sq = (xb * xb).sum(dim=1)

            # x · c^T  [bs, k]
            dots = self._dot(xb, centroids)

            # ||x - c||^2 = ||x||^2 - 2*x·c + ||c||^2
            dists = x_sq.unsqueeze(1) - 2 * dots + c_sq.unsqueeze(0)  # [bs, k]

            # Clamp to avoid negative due to float precision
            dists.clamp_(min=0.0)

            min_d, min_i = dists.min(dim=1)
            D_all[start:end] = min_d
            I_all[start:end] = min_i

        return D_all, I_all

    def _update_centroids(
        self, x: torch.Tensor, assignments: torch.Tensor, centroids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Compute new centroids as mean of assigned points.
        In distributed mode, aggregate sums and counts across ranks.
        
        Returns (new_centroids, counts, n_empty) where n_empty = number of empty clusters.
        """
        k, d = centroids.shape

        # Local sums and counts
        # Use scatter_add for efficiency
        counts = torch.zeros(k, device=self.device, dtype=torch.long)
        sums = torch.zeros(k, d, device=self.device, dtype=torch.float32)

        counts.scatter_add_(0, assignments, torch.ones_like(assignments))
        # Expand assignments for scatter_add on 2D
        idx = assignments.unsqueeze(1).expand(-1, d)
        sums.scatter_add_(0, idx, x)

        if self.distributed:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)

        # Find empty clusters
        empty_mask = counts == 0
        n_empty = int(empty_mask.sum().item())

        # Compute means (avoid div by zero)
        safe_counts = counts.clamp(min=1).float().unsqueeze(1)
        new_centroids = sums / safe_counts

        # Handle empty clusters: split the largest cluster
        if n_empty > 0:
            # Find the cluster with most points
            for empty_idx in torch.where(empty_mask)[0]:
                largest_idx = int(counts.argmax().item())
                # Split: perturb the largest centroid
                noise = torch.randn(d, device=self.device) * 1e-4
                new_centroids[empty_idx] = new_centroids[largest_idx] + noise
                new_centroids[largest_idx] = new_centroids[largest_idx] - noise
                # Update count tracking for subsequent empty clusters
                half = counts[largest_idx] // 2
                counts[empty_idx] = half
                counts[largest_idx] = counts[largest_idx] - half

        if self.spherical:
            new_centroids = new_centroids / new_centroids.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        return new_centroids, counts, int(n_empty)

    def _train_once(self, x: torch.Tensor, seed: int) -> Tuple[torch.Tensor, float]:
        """Single KMeans run (one restart)."""
        t0 = time.time()
        centroids = self._init_centroids(x, seed)

        self._log(f"  Preprocessing in {time.time() - t0:.2f} s")

        for it in range(self.niter):
            # E-step: assign
            t_search = time.time()
            D, assign_idx = self._assign_batch(x, centroids)
            search_time = time.time() - t_search

            # Compute objective (total distance)
            local_obj = D.sum()
            if self.distributed:
                dist.all_reduce(local_obj, op=dist.ReduceOp.SUM)
            obj = local_obj.item()

            # Compute imbalance factor: max_count / (N/k)
            local_n = torch.tensor([x.shape[0]], device=self.device, dtype=torch.long)
            if self.distributed:
                dist.all_reduce(local_n, op=dist.ReduceOp.SUM)
            total_n = local_n.item()

            counts = torch.zeros(self.k, device=self.device, dtype=torch.long)
            counts.scatter_add_(0, assign_idx, torch.ones_like(assign_idx))
            if self.distributed:
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            
            max_count = counts.max().item()
            ideal_count = total_n / self.k
            imbalance = max_count / ideal_count if ideal_count > 0 else 0.0

            # M-step: update centroids
            if not self.frozen_centroids:
                centroids, _, n_split = self._update_centroids(x, assign_idx, centroids)
            else:
                n_split = 0

            total_time = time.time() - t0

            self._log(
                f"  Iteration {it} ({total_time:.2f} s, search {search_time:.2f} s): "
                f"objective={obj:.0f} imbalance={imbalance:.3f} nsplit={n_split}"
            )

        # Final objective
        D_final, _ = self._assign_batch(x, centroids)
        local_obj = D_final.sum()
        if self.distributed:
            dist.all_reduce(local_obj, op=dist.ReduceOp.SUM)
        final_obj = local_obj.item()

        return centroids, final_obj

    def train(self, x: ArrayLike):
        """
        Train KMeans on data x.
        
        Parameters
        ----------
        x : np.ndarray or torch.Tensor of shape [N, d]
            Training data. In distributed mode, pass each rank's local shard.
        """
        x = self._to_tensor(x)
        assert x.shape[1] == self.d, f"Expected dim {self.d}, got {x.shape[1]}"

        # Subsample if needed
        x = self._subsample(x)

        n_local = x.shape[0]
        if self.distributed:
            total_n = torch.tensor([n_local], device=self.device, dtype=torch.long)
            dist.all_reduce(total_n, op=dist.ReduceOp.SUM)
            total_n = total_n.item()
        else:
            total_n = n_local

        self._log(
            f"Clustering {total_n} points in {self.d}D to {self.k} clusters, "
            f"redo {self.nredo} times, {self.niter} iterations"
        )
        if self.bf16 and not self._bf16_enabled:
            self._log("BF16 requested but unavailable on current device; falling back to FP32")
        if self._bf16_enabled:
            self._log("BF16 enabled for centroid distance matmul")

        best_centroids = None
        best_obj = float("inf")

        for redo in range(self.nredo):
            if self.nredo > 1:
                self._log(f"Redo {redo + 1}/{self.nredo}")

            seed = self.seed + redo * 12345
            centroids, obj = self._train_once(x, seed)

            if obj < best_obj:
                best_obj = obj
                best_centroids = centroids.clone()

        # Store as numpy (FAISS convention)
        assert best_centroids is not None
        self.centroids = best_centroids.cpu().numpy()
        self._log(f"Final objective: {best_obj:.0f}")

    def assign(self, x: ArrayLike, batch_size: int = 65536) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign points to nearest centroids.
        
        Handles arbitrarily large inputs by streaming chunks to GPU.
        
        Parameters
        ----------
        x : np.ndarray or torch.Tensor of shape [N, d]
        batch_size : int
            Number of vectors per GPU batch.
        
        Returns
        -------
        D : np.ndarray [N] — L2 distances to nearest centroid
        I : np.ndarray [N] — nearest centroid indices (int64)
        """
        assert self.centroids is not None, "Must call train() first"
        centroids = torch.from_numpy(self.centroids).to(self.device)
        c_sq = (centroids * centroids).sum(dim=1)  # [k]

        # Handle numpy input directly — stream to GPU in chunks
        if isinstance(x, np.ndarray):
            n = x.shape[0]
            D_all = np.empty(n, dtype=np.float32)
            I_all = np.empty(n, dtype=np.int64)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb = torch.from_numpy(x[start:end]).float().to(self.device)
                x_sq = (xb * xb).sum(dim=1)
                dots = self._dot(xb, centroids)
                dists = x_sq.unsqueeze(1) - 2 * dots + c_sq.unsqueeze(0)
                dists.clamp_(min=0.0)
                min_d, min_i = dists.min(dim=1)
                D_all[start:end] = min_d.cpu().numpy()
                I_all[start:end] = min_i.cpu().numpy()
            return D_all, I_all
        else:
            # Tensor path — move to device and batch
            x = x.float().to(self.device)
            D, assign_idx = self._assign_batch(x, centroids, batch_size=batch_size)
            return D.cpu().numpy(), assign_idx.cpu().numpy()
