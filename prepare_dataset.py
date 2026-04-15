"""
Prepare YOLO dataset — train/val split + dataset.yaml generation.

Usage:
    uv run python prepare_dataset.py
    uv run python prepare_dataset.py --split 0.8 --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Split data into YOLO train/val structure")
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw_images"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/labels/raw"))
    parser.add_argument("--split", type=float, default=0.8, help="Train fraction (default 0.8)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_dir: Path = args.images_dir
    labels_dir: Path = args.labels_dir

    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    # Collect images that have matching label files
    pairs = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))

    if not pairs:
        raise SystemExit("No image/label pairs found.")

    print(f"Found {len(pairs)} image/label pairs.")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * args.split)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Create output directories
    data_root = Path("data")
    dirs = {
        "train_img": data_root / "images" / "train",
        "train_lbl": data_root / "labels" / "train",
        "val_img": data_root / "images" / "val",
        "val_lbl": data_root / "labels" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Copy files
    for img_path, label_path in train_pairs:
        shutil.copy2(img_path, dirs["train_img"] / img_path.name)
        shutil.copy2(label_path, dirs["train_lbl"] / label_path.name)

    for img_path, label_path in val_pairs:
        shutil.copy2(img_path, dirs["val_img"] / img_path.name)
        shutil.copy2(label_path, dirs["val_lbl"] / label_path.name)

    # Generate dataset.yaml
    yaml_path = data_root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {data_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"  0: person\n"
    )

    print(f"Dataset YAML written to {yaml_path}")
    print("Done.")


if __name__ == "__main__":
    main()
