#!/usr/bin/env python3
"""
对多张 PNG 图片复现训练时的 Augmax 图像增强，并保存可视化结果。

示例：
python visualize_augmax.py \
    --images camera_0.png camera_1.png wrist.png \
    --output-dir augmax_vis \
    --num-aug 4 \
    --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import augmax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 JAX + Augmax 可视化图像增强效果。"
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="输入图片路径，可以一次传入三张或更多 PNG 图片。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("augmax_vis"),
        help="结果保存目录，默认：augmax_vis",
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=4,
        help="每张图片生成多少个随机增强结果，默认：4",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="JAX 随机种子，默认：42",
    )
    parser.add_argument(
        "--color-jitter-p",
        type=float,
        default=0.5,
        help=(
            "ColorJitter 的执行概率。原代码未指定 p，Augmax 默认值为 0.5；"
            "若想每张增强图都进行颜色扰动，可设为 1.0。"
        ),
    )
    return parser.parse_args()


def load_png(path: Path) -> jax.Array:
    """读取图片，返回 HWC、float32、范围 [-1, 1] 的 JAX 数组。"""
    with Image.open(path) as img:
        image_uint8 = np.asarray(img.convert("RGB"), dtype=np.uint8)

    image_01 = image_uint8.astype(np.float32) / 255.0
    image_m11 = image_01 * 2.0 - 1.0
    return jnp.asarray(image_m11)


def build_transform(
    image_m11: jax.Array,
    image_key: str,
    color_jitter_p: float,
) -> augmax.Chain:
    """
    根据原始代码构建增强流水线。

    非 wrist 相机：
        RandomCrop(95%) -> Resize -> Rotate(-5°, 5°) -> ColorJitter

    wrist 相机：
        ColorJitter
    """
    # 单张图片是 HWC，因此高度和宽度取 shape[:2]。
    # 原训练代码中的 image 是 BHWC，才会使用 shape[1:3]。
    height, width = map(int, image_m11.shape[:2])

    transforms = []

    if "wrist" not in image_key.lower():
        crop_width = max(1, int(width * 0.95))
        crop_height = max(1, int(height * 0.95))

        transforms.extend(
            [
                augmax.RandomCrop(crop_width, crop_height),
                augmax.Resize(width, height),
                augmax.Rotate((-5, 5)),
            ]
        )

    transforms.append(
        augmax.ColorJitter(
            brightness=0.3,
            contrast=0.4,
            saturation=0.5,
            p=color_jitter_p,
        )
    )

    return augmax.Chain(*transforms)


def augment_one(
    rng: jax.Array,
    image_m11: jax.Array,
    transform: augmax.Chain,
) -> np.ndarray:
    """
    完整复现：
        [-1, 1] -> [0, 1] -> Augmax -> [-1, 1]

    返回用于显示和保存的 [0, 1] NumPy 图像。
    """
    # Convert from [-1, 1] to [0, 1] for Augmax.
    image_01 = image_m11 / 2.0 + 0.5

    augmented_01 = transform(rng, image_01)

    # Back to [-1, 1].
    augmented_m11 = augmented_01 * 2.0 - 1.0

    # 可视化时再转回 [0, 1]。
    display_01 = augmented_m11 / 2.0 + 0.5
    display_01 = jnp.clip(display_01, 0.0, 1.0)

    return np.asarray(jax.device_get(display_01), dtype=np.float32)


def to_display(image_m11: jax.Array) -> np.ndarray:
    """将 [-1, 1] 图像转换为可显示的 [0, 1] NumPy 图像。"""
    image_01 = jnp.clip(image_m11 / 2.0 + 0.5, 0.0, 1.0)
    return np.asarray(jax.device_get(image_01), dtype=np.float32)


def save_image(image_01: np.ndarray, path: Path) -> None:
    """将 [0, 1] RGB 图像保存为 PNG。"""
    image_uint8 = np.clip(image_01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8, mode="RGB").save(path)


def save_single_comparison(
    original: np.ndarray,
    augmented_images: list[np.ndarray],
    title: str,
    output_path: Path,
) -> None:
    """保存单张输入图片的“原图 + 多个增强结果”对比图。"""
    images = [original, *augmented_images]
    labels = ["Original"] + [f"Augmented {i}" for i in range(1, len(images))]

    fig, axes = plt.subplots(
        1,
        len(images),
        figsize=(4 * len(images), 4),
        squeeze=False,
    )

    for ax, image, label in zip(axes[0], images, labels):
        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_all_comparison(
    rows: list[tuple[str, np.ndarray, list[np.ndarray]]],
    output_path: Path,
) -> None:
    """将所有输入图片排成总览图：每一行对应一张输入图片。"""
    if not rows:
        return

    num_cols = 1 + max(len(item[2]) for item in rows)
    fig, axes = plt.subplots(
        len(rows),
        num_cols,
        figsize=(4 * num_cols, 4 * len(rows)),
        squeeze=False,
    )

    for row_idx, (name, original, augmented_images) in enumerate(rows):
        row_images = [original, *augmented_images]

        for col_idx in range(num_cols):
            ax = axes[row_idx, col_idx]
            ax.axis("off")

            if col_idx < len(row_images):
                ax.imshow(row_images[col_idx])
                if col_idx == 0:
                    ax.set_title(f"{name}\nOriginal")
                else:
                    ax.set_title(f"Augmented {col_idx}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.num_aug < 1:
        raise ValueError("--num-aug 必须至少为 1。")

    if not 0.0 <= args.color_jitter_p <= 1.0:
        raise ValueError("--color-jitter-p 必须位于 [0, 1]。")

    image_paths = [Path(path) for path in args.images]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"以下图片不存在：{missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    master_rng = jax.random.PRNGKey(args.seed)
    comparison_rows: list[tuple[str, np.ndarray, list[np.ndarray]]] = []

    for image_index, image_path in enumerate(image_paths):
        image_m11 = load_png(image_path)
        original = to_display(image_m11)

        transform = build_transform(
            image_m11=image_m11,
            image_key=image_path.stem,
            color_jitter_p=args.color_jitter_p,
        )

        # 每张图片、每个增强样本都使用独立随机数。
        image_rng = jax.random.fold_in(master_rng, image_index)
        sub_rngs = jax.random.split(image_rng, args.num_aug)

        augmented_images = [
            augment_one(rng, image_m11, transform)
            for rng in sub_rngs
        ]

        image_output_dir = args.output_dir / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        save_image(original, image_output_dir / "original.png")

        for aug_index, augmented in enumerate(augmented_images, start=1):
            save_image(
                augmented,
                image_output_dir / f"augmented_{aug_index:02d}.png",
            )

        comparison_path = image_output_dir / "comparison.png"
        save_single_comparison(
            original=original,
            augmented_images=augmented_images,
            title=image_path.name,
            output_path=comparison_path,
        )

        comparison_rows.append(
            (image_path.name, original, augmented_images)
        )

        print(f"[完成] {image_path}")
        print(f"       结果目录：{image_output_dir}")
        print(f"       对比图：  {comparison_path}")

    all_comparison_path = args.output_dir / "all_images_comparison.png"
    save_all_comparison(comparison_rows, all_comparison_path)

    print(f"\n全部完成，总览图：{all_comparison_path}")


if __name__ == "__main__":
    main()
