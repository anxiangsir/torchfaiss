# FAISS Retest Evaluation Report (1× + 20×)

## Overview

This report summarizes a fresh retest of the original FAISS multi-GPU KMeans path and compares it against the latest TorchFAISS full-data baseline.

Test environment (same project defaults):

- Hardware: 8× NVIDIA A800-SXM4-80GB
- Features: ImageNet CLIP embeddings (d=768, unit-normalized)
- KMeans config: `k=1000`, `niter=20`, `seed=1234`

## Commands Used

```bash
# FAISS 1× retest
python benchmark_faiss.py --ngpu 8 --feature_dir ./features --result_dir ./results_faiss_retest

# FAISS 20× retest
python benchmark_faiss_20x.py --ngpu 8 --feature_dir ./features_20x --result_dir ./results_20x_faiss_retest

# Evaluation artifacts
python compare_results.py --result_dir ./results_faiss_retest
python compare_results.py --result_dir ./results_20x_faiss_retest --feature_dir ./features_20x
```

## Raw Retest Results

### 1× (FAISS retest)

Source: `results_faiss_retest/faiss_result.json`

- Train time: **17.35 s**
- Assign time: **0.32 s** (train), **0.04 s** (val)
- Val NMI: **0.4591**
- Val Purity: **0.0179**

### 20× (FAISS retest)

Source: `results_20x_faiss_retest/faiss_result.json`

- Train time: **72.36 s**
- Assign time: **5.88 s** (train), **0.10 s** (val)
- Val NMI: **0.0003**
- Val Purity: **0.0010**

## Quality Diagnostics from Evaluation Artifacts

### 1× diagnostic highlights

Source: `results_faiss_retest/evaluation_reports/faiss_8gpu/detailed_metrics.json`

- Non-empty clusters: **2 / 998**
- Top-1 accuracy (cluster→label mapping): **0.0010**
- ROC-AUC (macro): **0.5000** (random-level)

### 20× diagnostic highlights

Source: `results_20x_faiss_retest/evaluation_reports/faiss_8gpu/detailed_metrics.json`

- Non-empty clusters: **134 / 998**
- Top-1 accuracy (cluster→label mapping): **0.0010**
- ROC-AUC (macro): **0.5000** (random-level)

These indicators are consistent with strong cluster collapse / degeneracy on this unit-normalized feature distribution under FAISS default training behavior.

## Comparison with Latest TorchFAISS Full-Data Baseline (20×)

TorchFAISS reference source: `results_20x_full_fp32_notriton_after_bf16cache/torchfaiss_result.json`

| Method | Train (s) | Assign-Train (s) | Val NMI | Val Purity |
|---|---:|---:|---:|---:|
| FAISS retest (8GPU) | 72.36 | **5.88** | 0.0003 | 0.0010 |
| TorchFAISS full-data FP32 (8GPU) | **12.92** | 58.39 | **0.7935** | **0.5415** |

Key readout:

- **Train phase**: TorchFAISS is much faster on full-data training and converges to meaningful clusters.
- **Assign phase**: FAISS assign on learned centroids is fast, but quality is near random due to degraded centroids.
- **End quality**: TorchFAISS is decisively better on NMI/purity.

## Conclusion

The FAISS retest reproduces the same failure mode observed previously on this dataset family: fast assignment but poor clustering quality due to unstable/degenerated centroid training dynamics. For this workload and metric target, TorchFAISS remains the reliable choice.
