"""
Extract CLIP ViT-L/14 features for ImageNet train/val sets.
Uses torchrun for distributed extraction across GPUs.

Usage:
    torchrun --nproc_per_node=8 extract_features.py
"""

import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder
import clip
from tqdm import tqdm

from torch.cuda.amp import autocast


def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Extract CLIP features for ImageNet")
    parser.add_argument("--data_root", type=str, default="./imagenet")
    parser.add_argument("--output_dir", type=str, default=os.path.join(repo_dir, "features"))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    # Init distributed (torchrun sets env vars)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Extracting CLIP ViT-L/14 features using {world_size} GPUs")
        print(f"Data root: {args.data_root}")
        print(f"Output dir: {args.output_dir}")

    # Load CLIP model
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    pbar = None
    for split in args.splits:
        data_path = os.path.join(args.data_root, split)
        dataset = ImageFolder(data_path, transform=preprocess)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        all_features = []
        all_labels = []

        if rank == 0:
            pbar = tqdm(total=len(loader), desc=f"[{split}] Extracting features")

        with torch.no_grad(), autocast():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                features = model.encode_image(images)
                features = features.float()
                # L2 normalize (standard for CLIP)
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features.cpu())
                all_labels.append(labels)
                if rank == 0 and pbar is not None:
                    pbar.update(1)

        if rank == 0 and pbar is not None:
            pbar.close()

        local_features = torch.cat(all_features, dim=0)  # [local_N, 768]
        local_labels = torch.cat(all_labels, dim=0)       # [local_N]

        # Gather all features to rank 0
        local_size = torch.tensor([local_features.shape[0]], dtype=torch.long, device=device)
        all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        all_sizes_list = [int(s.item()) for s in all_sizes]
        max_size = int(max(all_sizes_list))

        # Pad for uniform all_gather
        pad_features = torch.zeros((max_size, local_features.shape[1]), device=device)
        pad_features[:local_features.shape[0]] = local_features.to(device)
        pad_labels = torch.full((max_size,), -1, dtype=torch.long, device=device)
        pad_labels[:local_labels.shape[0]] = local_labels.to(device)

        gathered_features = [torch.zeros_like(pad_features) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(pad_labels) for _ in range(world_size)]
        dist.all_gather(gathered_features, pad_features)
        dist.all_gather(gathered_labels, pad_labels)

        if rank == 0:
            features_list = []
            labels_list = []
            for i in range(world_size):
                n = all_sizes_list[i]
                features_list.append(gathered_features[i][:n].cpu())
                labels_list.append(gathered_labels[i][:n].cpu())

            final_features = torch.cat(features_list, dim=0).numpy()
            final_labels = torch.cat(labels_list, dim=0).numpy()

            os.makedirs(args.output_dir, exist_ok=True)
            feat_path = os.path.join(args.output_dir, f"{split}_features.npy")
            label_path = os.path.join(args.output_dir, f"{split}_labels.npy")
            np.save(feat_path, final_features)
            np.save(label_path, final_labels)
            print(f"[{split}] Saved features: {final_features.shape} -> {feat_path}")
            print(f"[{split}] Saved labels:   {final_labels.shape} -> {label_path}")

        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
