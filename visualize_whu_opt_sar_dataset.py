import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from datasets import WHUOptSarPatchDataset


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


def make_optical_rgb(optical: np.ndarray) -> np.ndarray:
    if optical.ndim != 3:
        raise ValueError(f"optical must be (C, H, W), got {optical.shape}.")
    if optical.shape[0] < 3:
        raise ValueError(f"optical needs at least 3 channels, got {optical.shape[0]}.")

    # 数据中前三个通道按 B, G, R 解释，显示时转换为 RGB。
    blue = minmax01(optical[0])
    green = minmax01(optical[1])
    red = minmax01(optical[2])
    return np.stack([red, green, blue], axis=-1)


def make_sar_views(sar: np.ndarray):
    if sar.ndim != 3:
        raise ValueError(f"sar must be (C, H, W), got {sar.shape}.")
    if sar.shape[0] == 1:
        return [minmax01(sar[0])], 1
    if sar.shape[0] >= 2:
        return [minmax01(sar[0]), minmax01(sar[1])], sar.shape[0]
    raise ValueError(f"sar has invalid channel count: {sar.shape[0]}")


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
    ]
    cmap = ListedColormap(colors, name="whu_labels")
    cmap.set_bad(color="black")
    return cmap


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize WHUOptSarPatchDataset samples.")
    parser.add_argument("--root_dir", type=str, required=True, help="WHU dataset root directory.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Dataset split.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio used by the dataset.")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size used by the dataset.")
    parser.add_argument("--output_size", type=int, default=None, help="Center-cropped output size.")
    parser.add_argument("--index", type=int, default=None, help="Sample index to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for index selection.")
    parser.add_argument("--no_normalize", action="store_true", help="Disable dataset normalization for visualization.")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save the figure.")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = WHUOptSarPatchDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        patch_size=args.patch_size,
        output_size=args.output_size,
        normalize=not args.no_normalize,
    )

    if len(dataset) == 0:
        raise ValueError("The dataset is empty. Check the split, patch_size, stride_ratio, and drop settings.")

    random.seed(args.seed)
    index = args.index if args.index is not None else random.randint(0, len(dataset) - 1)
    optical, sar, label = dataset[index]

    optical_np = to_numpy(optical)
    sar_np = to_numpy(sar)
    label_np = to_numpy(label)

    optical_rgb = make_optical_rgb(optical_np)
    sar_views, sar_channels = make_sar_views(sar_np)
    masked_label = np.ma.masked_equal(label_np, 0)
    label_cmap = make_label_cmap()

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), constrained_layout=True)

    axes[0].imshow(optical_rgb)
    axes[0].set_title("Optical RGB (B,G,R -> RGB)")
    axes[0].axis("off")

    axes[1].imshow(sar_views[0], cmap="gray")
    axes[1].set_title("SAR Channel 1")
    axes[1].axis("off")

    if len(sar_views) >= 2:
        axes[2].imshow(sar_views[1], cmap="gray")
        axes[2].set_title("SAR Channel 2")
    else:
        axes[2].text(0.5, 0.5, "SAR Channel 2\nnot available", ha="center", va="center", fontsize=12)
        axes[2].set_title("SAR Channel 2")
    axes[2].axis("off")

    axes[3].imshow(masked_label, cmap=label_cmap, vmin=0, vmax=7, interpolation="nearest")
    axes[3].set_title("Label")
    axes[3].axis("off")

    fig.suptitle(
        f"WHUOptSarPatchDataset | split={args.split} | index={index} | SAR channels={sar_channels}",
        fontsize=14,
    )

    if args.save_path:
        fig.savefig(args.save_path, dpi=200, bbox_inches="tight")
        print(f"Saved visualization to: {args.save_path}")

    plt.show()


if __name__ == "__main__":
    main()