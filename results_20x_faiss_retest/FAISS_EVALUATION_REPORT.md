# FAISS 20× Retest Evaluation Report

## Setup

- Dataset: `features_20x` (25,623,360 train / 1,000,000 val, d=768)
- Method: FAISS KMeans multi-GPU (`ngpu=8`)
- Params: `k=1000`, `niter=20`, `seed=1234`

Command:

```bash
python benchmark_faiss_20x.py --ngpu 8 --feature_dir ./features_20x --result_dir ./results_20x_faiss_retest
python compare_results.py --result_dir ./results_20x_faiss_retest --feature_dir ./features_20x
```

## Result Snapshot

From `faiss_result.json`:

- Train time: **72.364 s**
- Assign train time: **5.880 s**
- Assign val time: **0.095 s**
- Val NMI: **0.0003**
- Val Purity: **0.0010**

From `evaluation_reports/faiss_8gpu/detailed_metrics.json`:

- Non-empty clusters: **134 / 998**
- Top-1 accuracy (cluster→label): **0.0010**
- Top-5 accuracy: **0.0050**
- ROC-AUC macro: **0.5000**

## Interpretation

This run shows a high-speed assignment stage but near-random downstream quality indicators, suggesting cluster degeneration on this feature distribution with the current FAISS training path/configuration.

For end-to-end clustering quality on this workload, compare against TorchFAISS full-data baseline:

- `results_20x_full_fp32_notriton_after_bf16cache/torchfaiss_result.json`
