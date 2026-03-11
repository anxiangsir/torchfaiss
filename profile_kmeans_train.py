import argparse
import json
import os
import time
from typing import Dict, Any

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
from torchprofile import profile_macs

from torchfaiss import TorchKmeans


class DotWrapper(torch.nn.Module):
    def __init__(self, km: TorchKmeans, centroids: torch.Tensor):
        super().__init__()
        self.km = km
        self.register_buffer("centroids", centroids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.km._dot(x, self.centroids)


def run_case(
    train_features: np.ndarray,
    k: int,
    niter: int,
    mode: str,
    sample_size: int,
    seed: int,
    out_dir: str,
) -> Dict[str, Any]:
    use_bf16 = mode == "bf16"
    use_int8 = mode == "int8"

    rng = np.random.default_rng(seed)
    idx = rng.choice(train_features.shape[0], size=min(sample_size, train_features.shape[0]), replace=False)
    x = train_features[idx].astype(np.float32)

    km = TorchKmeans(
        d=x.shape[1],
        k=k,
        niter=niter,
        gpu=True,
        distributed=False,
        verbose=False,
        seed=seed,
        bf16=use_bf16,
        int8_assign=use_int8,
        max_points_per_centroid=max(1, sample_size // k + 1),
    )

    x_t = torch.from_numpy(x).to(km.device)
    centroids = km._init_centroids(x_t, seed)

    dot_module = DotWrapper(km, centroids)
    with torch.no_grad():
        macs = profile_macs(dot_module, args=(x_t,))

    per_iter = []
    with torch.no_grad():
        for _ in range(niter):
            torch.cuda.synchronize(km.device)
            t0 = time.time()
            dists, assigns = km._assign_batch(x_t, centroids, batch_size=65536)
            torch.cuda.synchronize(km.device)
            t_assign = time.time() - t0

            torch.cuda.synchronize(km.device)
            t1 = time.time()
            centroids, _, _ = km._update_centroids(x_t, assigns, centroids)
            torch.cuda.synchronize(km.device)
            t_update = time.time() - t1
            per_iter.append({"assign_s": t_assign, "update_s": t_update})

    trace_path = os.path.join(out_dir, f"trace_{mode}.json")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            dists, assigns = km._assign_batch(x_t, centroids, batch_size=65536)
            _ = km._update_centroids(x_t, assigns, centroids)

    prof.export_chrome_trace(trace_path)
    key_table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)

    return {
        "mode": mode,
        "sample_size": int(x.shape[0]),
        "k": k,
        "d": int(x.shape[1]),
        "niter": niter,
        "dot_macs": int(macs),
        "dot_flops_est": int(2 * macs),
        "per_iter": per_iter,
        "mean_assign_s": float(np.mean([it["assign_s"] for it in per_iter])),
        "mean_update_s": float(np.mean([it["update_s"] for it in per_iter])),
        "trace_path": trace_path,
        "profiler_top": key_table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default="./features_20x")
    parser.add_argument("--output", type=str, default="./results/profile_kmeans_train.json")
    parser.add_argument("--sample_size", type=int, default=320000)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--niter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    train_features = np.load(os.path.join(args.feature_dir, "train_features.npy"), mmap_mode="r")

    results = []
    for mode in ("fp32", "bf16"):
        print(f"Profiling mode={mode}", flush=True)
        results.append(
            run_case(
                train_features=train_features,
                k=args.k,
                niter=args.niter,
                mode=mode,
                sample_size=args.sample_size,
                seed=args.seed,
                out_dir=os.path.dirname(os.path.abspath(args.output)),
            )
        )

    summary = {"config": vars(args), "results": results}
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "config": summary["config"],
        "results": [
            {
                "mode": r["mode"],
                "mean_assign_s": r["mean_assign_s"],
                "mean_update_s": r["mean_update_s"],
                "dot_flops_est": r["dot_flops_est"],
                "trace_path": r["trace_path"],
            }
            for r in results
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
