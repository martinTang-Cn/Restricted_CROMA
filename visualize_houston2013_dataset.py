import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from datasets import Houston2013PatchDataset


def to_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def minmax01(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    min_value = np.nanmin(image)
    max_value = np.nanmax(image)
    if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value <= min_value:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def make_hsi_rgb(hsi: np.ndarray) -> np.ndarray:
    if hsi.ndim != 3:
        raise ValueError(f"HSI must be (C, H, W), got {hsi.shape}.")

    channel_indices = [67, 31, 14]
    if hsi.shape[0] <= max(channel_indices):
        raise ValueError(
            f"HSI has only {hsi.shape[0]} channels, cannot index {channel_indices}."
        )

    rgb = np.stack([minmax01(hsi[idx]) for idx in channel_indices], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def make_lidar_gray(lidar: np.ndarray) -> np.ndarray:
    if lidar.ndim == 3:
        if lidar.shape[0] != 1:
            raise ValueError(f"LiDAR must be (1, H, W), got {lidar.shape}.")
        lidar = lidar[0]
    elif lidar.ndim != 2:
        raise ValueError(f"LiDAR must be (H, W) or (1, H, W), got {lidar.shape}.")
    return minmax01(lidar)


def make_label_cmap():
    colors = [
        "#000000",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
    ]
    return ListedColormap(colors[:15], name="houston2013_label")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Houston2013PatchDataset samples.")
    parser.add_argument("--root_dir", type=str, required=True, help="Houston2013 dataset root directory.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Dataset split.")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size used by the dataset.")
    parser.add_argument("--output_size", type=int, default=None, help="Cropped output size after transform.")
    parser.add_argument("--index", type=int, default=None, help="Sample index to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for index selection.")
    parser.add_argument("--no_normalize", action="store_true", help="Disable dataset normalization for visualization.")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save the figure.")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = Houston2013PatchDataset(
        root_dir=args.root_dir,
        split=args.split,
        patch_size=args.patch_size,
        output_size=args.output_size,
        normalize=not args.no_normalize,
    )

    if len(dataset) == 0:
        raise ValueError("The dataset is empty. Check the split, patch_size, and drop_empty settings.")

    random.seed(args.seed)
    index = args.index if args.index is not None else random.randint(0, len(dataset) - 1)
    hsi, lidar, label = dataset[index]

    hsi_np = to_numpy(hsi)
    lidar_np = to_numpy(lidar)
    label_np = to_numpy(label)

    hsi_rgb = make_hsi_rgb(hsi_np)
    lidar_gray = make_lidar_gray(lidar_np)
    masked_label = np.ma.masked_equal(label_np, -1)
    label_cmap = make_label_cmap()
    label_cmap.set_bad(color="black")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    axes[0].imshow(hsi_rgb)
    axes[0].set_title("HSI RGB (68, 32, 15)")
    axes[0].axis("off")

    axes[1].imshow(lidar_gray, cmap="gray")
    axes[1].set_title("LiDAR")
    axes[1].axis("off")

    axes[2].imshow(masked_label, cmap=label_cmap, vmin=0, vmax=14, interpolation="nearest")
    axes[2].set_title("Label")
    axes[2].axis("off")

    fig.suptitle(f"Houston2013PatchDataset | split={args.split} | index={index}", fontsize=14)

    if args.save_path:
        fig.savefig(args.save_path, dpi=200, bbox_inches="tight")
        print(f"Saved visualization to: {args.save_path}")

    plt.show()


if __name__ == "__main__":
    main()