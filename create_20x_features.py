import argparse
import os

import numpy as np


def main() -> None:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default=os.path.join(repo_dir, "features"))
    parser.add_argument("--dst_dir", type=str, default=os.path.join(repo_dir, "features_20x"))
    parser.add_argument("--scale", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    for split in ["train", "val"]:
        feat_path = os.path.join(args.src_dir, f"{split}_features.npy")
        label_path = os.path.join(args.src_dir, f"{split}_labels.npy")
        features = np.load(feat_path)
        labels = np.load(label_path)

        features_scaled = np.tile(features, (args.scale, 1))
        labels_scaled = np.tile(labels, args.scale)

        np.save(os.path.join(args.dst_dir, f"{split}_features.npy"), features_scaled)
        np.save(os.path.join(args.dst_dir, f"{split}_labels.npy"), labels_scaled)

        print(
            f"[{split}] {features.shape} -> {features_scaled.shape}; "
            f"labels {labels.shape} -> {labels_scaled.shape}"
        )

    print(f"Saved scaled features to: {args.dst_dir}")


if __name__ == "__main__":
    main()
