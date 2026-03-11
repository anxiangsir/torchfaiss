# TorchFAISS

<p align="center">
  <img src="assets/logo.svg" alt="TorchFAISS logo" width="900"/>
</p>

Pure PyTorch distributed KMeans implementation with a FAISS-compatible API (`train` / `assign` / `centroids`) and multi-GPU / multi-node training via `torch.distributed`.

GitHub Pages: https://anxiangsir.github.io/torchfaiss/

## Features

- FAISS-compatible API for fast migration
- Data-parallel distributed training with NCCL all-reduce
- Streaming assignment for large datasets
- Optional 20× scaled benchmark workflow
- Rich post-hoc evaluation against original IN1K labels (P/R/F1, ROC/PR, confusion matrix, cluster diagnostics)

## Installation

```bash
pip install -r requirements.txt
```

Install directly from GitHub (pip install git+):

```bash
pip install "git+https://github.com/anxiangsir/torchfaiss.git"

# if your environment uses a restricted mirror, use:
pip install --no-build-isolation "git+https://github.com/anxiangsir/torchfaiss.git"
```

Install a specific branch/tag/commit:

```bash
# branch
pip install "git+https://github.com/anxiangsir/torchfaiss.git@main"

# tag
pip install "git+https://github.com/anxiangsir/torchfaiss.git@v0.1.0"

# commit
pip install "git+https://github.com/anxiangsir/torchfaiss.git@<commit_sha>"
```

> Note: editable install (`pip install -e .`) depends on newer build-backend support in your pip/setuptools environment. Standard install (`pip install .`) and wheel install work reliably.

Build and install wheel (.whl):

```bash
python -m pip install --upgrade build
python -m build --no-isolation

# install built wheel
pip install dist/torchfaiss-0.1.0-py3-none-any.whl
```

## Quick Start

```python
from torchfaiss import TorchKmeans

km = TorchKmeans(d=768, k=1000, niter=20, gpu=True, verbose=True)
km.train(x_train)
D, I = km.assign(x_test)
```

Distributed:

```python
import torch.distributed as dist
from torchfaiss import TorchKmeans

dist.init_process_group(backend="nccl")
km = TorchKmeans(d=768, k=1000, niter=20, distributed=True, verbose=True)
km.train(x_local_shard)
D, I = km.assign(x_test)
```

BF16 (optional, CUDA device must support BF16):

```python
km = TorchKmeans(d=768, k=1000, niter=20, distributed=True, bf16=True)
```

`bf16=True` uses BF16 for the distance matmul hot path and keeps accumulations/objective in FP32 for stability.

Triton compile path + optional INT8 assignment (both optional):

```python
km = TorchKmeans(
    d=768,
    k=1000,
    niter=20,
    distributed=True,
    use_triton=True,
    int8_assign=True,
)
```

- `use_triton=True`: enables a Triton-backed `torch.compile` distance path on CUDA when Triton is available.
- `int8_assign=True`: enables optional INT8 assignment matmul path (falls back automatically when unsupported).

## Why TorchFAISS Is Faster Than FAISS Here

TorchFAISS uses distributed data parallelism: each of N GPUs processes 1/N of the data per Lloyd iteration,
communicating only centroid statistics (O(K·d)) instead of all data. This is fundamentally different from
FAISS multi-GPU, which replicates the full dataset onto every GPU and shards only the index queries.

1. **Assignment/update compute is split across GPUs**. Each rank handles ~N/world_size samples, so most O(N·K·d) work scales with GPU count.
2. **Per-iteration communication is relatively small**. Synchronization is over centroid statistics (roughly O(K·d)), while local distance compute stays O(N·K·d); as N grows, compute dominates and communication overhead is amortized.
3. **Better scaling at larger N**. Speedup rises with dataset size, consistent with communication overhead becoming less significant.
4. **Where speedup is not guaranteed**: for small datasets or communication-heavy settings, distributed overhead can offset gains.

In short: data-parallel sharding fundamentally differs from FAISS's replicated multi-GPU approach — and this difference is what drives speedup at scale.

## TorchFAISS Design Philosophy

TorchFAISS is designed around three principles:

1. **FAISS-compatible surface, PyTorch-native internals**
   - Keep migration cost low (`train/assign/centroids` interface)
   - Use pure PyTorch + `torch.distributed`, no custom C++/CUDA extension burden

2. **Scale-first KMeans implementation**
   - Treat assignment as the dominant O(N·K·d) workload
   - Prioritize sharding, batched compute, and communication-efficient centroid updates

3. **Performance with numerical stability**
   - Support optional BF16 in matmul hot path for speed
   - Keep key accumulations/objective paths in FP32 for stable convergence behavior

## Optimization Techniques in This Repo

- **Distributed data-parallel Lloyd iterations**: each rank computes local assignment/statistics, then all-reduce for centroid updates.
- **Batched nearest-centroid assignment**: avoids OOM and keeps GPU compute saturated.
- **Distance identity optimization** (`||x-c||^2 = ||x||^2 - 2x·c + ||c||^2`): reduces redundant compute.
- **Streaming assignment for large datasets**: chunk-based assignment on full train/val for 20× scale.
- **Empty-cluster repair strategy**: split-largest-cluster style fallback to keep K fixed and training stable.
- **Optional BF16 hot-path acceleration**: faster centroid-distance matmul when hardware supports BF16.
- **Optional Triton compile acceleration**: enables compiled distance path on CUDA.
- **Optional INT8 assignment acceleration**: INT8 GEMM-based assignment path with safe runtime fallback.
- **Portable path defaults**: scripts default to repo-relative paths for reproducibility across machines.

## Benchmark Summary

Hardware: 8× NVIDIA A800-SXM4-80GB. Data: ImageNet CLIP features (d=768, unit-normalized). k=1000, niter=20, seed=1234.
Both methods use 8 GPUs. Train time only (excludes post-hoc assignment).

### 1× ImageNet Features (1.28M train vectors)

| Method | Train Time | Speedup | Val NMI |
|---|---:|---:|---:|
| FAISS (8GPU, `gpu=8`) | 10.66 s | 1.0× | 0.459 (degenerate) |
| TorchFAISS (8GPU, `torchrun`) | **0.50 s** | **21.3×** | **0.787** |

### 20× ImageNet Features (25.6M train vectors)

| Method | Train Time | Speedup | Val NMI |
|---|---:|---:|---:|
| FAISS (8GPU, `gpu=8`) | 55.62 s | 1.0× | 0.000 (degenerate) |
| TorchFAISS (8GPU, `torchrun`) | **10.01 s** | **5.6×** | **0.793** |

### 20× TorchFAISS Ablation (No Triton): FP32 vs INT8 Assign

Measured with:

- `torchrun --nproc_per_node=8 benchmark_20x.py --feature_dir ./features_20x --result_dir ./results_20x_fp32_notriton`
- `torchrun --nproc_per_node=8 benchmark_20x.py --feature_dir ./features_20x --result_dir ./results_20x_int8_notriton --int8_assign`

| TorchFAISS Mode | Train Time | Assign Time (Train) | Assign Time (Val) | Val NMI | Val Purity | Assign Peak Mem |
|---|---:|---:|---:|---:|---:|---:|
| FP32 (no Triton) | **6.96 s** | 167.78 s | **0.53 s** | 0.7890 | 0.5251 | **1204.1 MB** |
| INT8 Assign (no Triton) | 8.99 s | **68.58 s** | 0.62 s | **0.7917** | **0.5265** | 1252.8 MB |

Observed trade-off on 20×:

- INT8 assign significantly accelerates full-train assignment (~59% faster)
- INT8 training pass is slower in this setup (~29% slower)
- Validation clustering quality remains close (slightly higher NMI/purity in this run)

> **Why FAISS degenerates on this dataset**: These CLIP features are unit-normalized (L2 norm ≈ 1.0). FAISS
> subsamples to 256K points and initializes centroids randomly from that subsample. On a unit hypersphere
> with d=768, random init leads to all points being equidistant from all centroids, causing every cluster
> to empty out and be re-split every iteration (`imbalance=1000, nsplit=999`). TorchFAISS avoids this by
> sharding the full dataset across 8 GPUs — each GPU draws a stratified 32K subsample, yielding better
> coverage of the hypersphere and stable convergence (imbalance ≈ 4, no splits after iteration 0).
>
> **Why the speedup grows at 20×**: At 20× FAISS still trains on only 256K points (its hard subsample cap),
> but its preprocessing + load time balloons to ~47s. TorchFAISS trains on the full 25.6M sharded data
> in 10s because compute scales linearly with GPUs while communication overhead stays fixed at O(K·d).
## Enhanced Evaluation vs Original IN1K Labels

`compare_results.py` now evaluates saved clustering outputs against original IN1K labels and generates:

- Clustering metrics: NMI, AMI, ARI, Rand, Fowlkes-Mallows, homogeneity, completeness, V-measure, purity
- Classification-style metrics from cluster→label mapping: accuracy, precision/recall/F1 (macro/micro/weighted), top-1/top-5
- Curves: one-vs-rest ROC and PR (micro/macro + top classes)
- Figures: confusion matrix (top classes), cluster size curve, cluster purity histogram

Run:

```bash
python compare_results.py --result_dir ./results
python compare_results.py --result_dir ./results_20x
```

Artifacts are saved under:

- `results/evaluation_reports/`
- `results_20x/evaluation_reports/`

### Speed Comparison Figures

1× dataset:

![1x Speed Comparison](results/evaluation_reports/speed_comparison.png)

20× dataset:

![20x Speed Comparison](results_20x/evaluation_reports/speed_comparison.png)

### Unified FAISS vs TorchFAISS Curve Comparison (same figure)

The comparison image now includes both **full-range** and **zoomed** panels (low-FPR ROC and high-recall PR) so small quality gaps are easier to see.

1× dataset:

![1x Curve Comparison](results/evaluation_reports/curve_comparison.png)

20× dataset:

![20x Curve Comparison](results_20x/evaluation_reports/curve_comparison.png)

## Reproducing the Full Pipeline

```bash
# 1) Extract CLIP features (set your ImageNet root)
torchrun --nproc_per_node=8 extract_features.py --data_root ./imagenet --output_dir ./features

# 2) Build 20× scaled dataset
python create_20x_features.py --src_dir ./features --dst_dir ./features_20x --scale 20

# 3) TorchFAISS benchmarks
torchrun --nproc_per_node=8 benchmark.py --feature_dir ./features --result_dir ./results
torchrun --nproc_per_node=8 benchmark_20x.py --feature_dir ./features_20x --result_dir ./results_20x

# optional BF16 speed mode
torchrun --nproc_per_node=8 benchmark.py --feature_dir ./features --result_dir ./results --bf16
torchrun --nproc_per_node=8 benchmark_20x.py --feature_dir ./features_20x --result_dir ./results_20x --bf16

# optional Triton + INT8 assignment mode
torchrun --nproc_per_node=8 benchmark.py --feature_dir ./features --result_dir ./results --use_triton --int8_assign
torchrun --nproc_per_node=8 benchmark_20x.py --feature_dir ./features_20x --result_dir ./results_20x --use_triton --int8_assign

# 20x no-triton ablation (FP32 vs INT8 assign)
torchrun --nproc_per_node=8 benchmark_20x.py --feature_dir ./features_20x --result_dir ./results_20x_fp32_notriton
torchrun --nproc_per_node=8 benchmark_20x.py --feature_dir ./features_20x --result_dir ./results_20x_int8_notriton --int8_assign

# precision/speed/memory comparison across modes (single GPU)
python benchmark_precision_modes.py --feature_dir ./features --output ./results/precision_modes.json

# 4) FAISS benchmarks (run in env with faiss installed)
python benchmark_faiss.py --ngpu 8 --feature_dir ./features --result_dir ./results
python benchmark_faiss_20x.py --ngpu 8 --feature_dir ./features_20x --result_dir ./results_20x

# 5) Enhanced comparison + plotting
python compare_results.py --result_dir ./results
python compare_results.py --result_dir ./results_20x
```

## Project Structure

```text
torchfaiss/
├── torchfaiss/
│   ├── __init__.py
│   └── kmeans.py
├── extract_features.py
├── create_20x_features.py
├── benchmark.py
├── benchmark_faiss.py
├── benchmark_20x.py
├── benchmark_faiss_20x.py
├── compare_results.py
├── eval_utils.py
├── requirements.txt
└── README.md
```

## Citation

If you use this project in academic work, please cite:

```bibtex
@software{anxiangsir_torchfaiss_2026,
  author  = {anxiangsir},
  title   = {TorchFAISS: Pure PyTorch Distributed KMeans with a FAISS-Compatible API},
  year    = {2026},
  url     = {https://github.com/anxiangsir/torchfaiss}
}
```
