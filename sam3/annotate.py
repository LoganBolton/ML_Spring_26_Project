"""
Batch SAM3 annotation — generates YOLO-format bounding box labels for "person".

Usage:
    uv run python sam3/annotate.py
    uv run python sam3/annotate.py --input-dir data/raw_images --output-dir data/labels/raw --conf 0.25
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "sam3"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_ID = 0  # person
PROMPT = "person"


def xyxy_to_yolo(box: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert [x1, y1, x2, y2] pixel coords to YOLO normalized [x_center, y_center, w, h]."""
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return x_center, y_center, w, h


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate YOLO labels with SAM3")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/labels/raw"))
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if not CHECKPOINT.exists():
        raise SystemExit(f"Checkpoint not found: {CHECKPOINT}\nRun sam3/setup.sh first.")

    images = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"No images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}")
    print("Loading SAM3 model...")
    processor = Sam3Processor.from_pretrained(str(CHECKPOINT))
    model = Sam3Model.from_pretrained(str(CHECKPOINT), torch_dtype=dtype).to(device)
    model.eval()

    print(f"Processing {len(images)} images → {output_dir}/\n")

    total_boxes = 0
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=args.conf
        )[0]

        boxes = results["boxes"].cpu().tolist()

        # Write YOLO label file
        label_path = output_dir / f"{img_path.stem}.txt"
        with open(label_path, "w") as f:
            for box in boxes:
                xc, yc, bw, bh = xyxy_to_yolo(box, w, h)
                f.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        total_boxes += len(boxes)
        print(f"  {img_path.name}: {len(boxes)} box(es)")

    print(f"\nDone. {total_boxes} total boxes across {len(images)} images.")


if __name__ == "__main__":
    main()
