"""
TorchFAISS - A pure PyTorch distributed KMeans implementation.
API-compatible with faiss.Kmeans.

Supports:
  - Single GPU / CPU
  - Multi-GPU (torch.distributed, single-node or multi-node)
  - All FAISS Kmeans parameters: niter, nredo, spherical, seed, 
    max_points_per_centroid, min_points_per_centroid, frozen_centroids, verbose
"""

from .kmeans import TorchKmeans

__all__ = ["TorchKmeans"]
__version__ = "0.1.0"
