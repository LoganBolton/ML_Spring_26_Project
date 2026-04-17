"""
Train YOLO on SAM3-generated person annotations.

Usage:
    uv run python train.py
    uv run python train.py --epochs 50 --imgsz 640 --batch 16
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on person dataset")
    parser.add_argument("--data", type=str, default="data/dataset.yaml")
    parser.add_argument("--model", type=str, default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
