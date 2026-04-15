"""
SAM 3 Person Detection — Bounding Boxes
Runs SAM 3 on data/test_toomers.png and outputs bounding boxes for the "person" class.
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

# ---------- config ----------
IMAGE_PATH = Path(__file__).resolve().parent.parent / "data" / "test_toomers.png"
CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "sam3"
PROMPT = "person"
CONF_THRESHOLD = 0.25
# -----------------------------


def main() -> None:
    if not IMAGE_PATH.exists():
        sys.exit(f"Image not found: {IMAGE_PATH}")
    if not CHECKPOINT.exists():
        sys.exit(f"Checkpoint not found: {CHECKPOINT}\nRun setup.sh first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}")

    # Load model + processor from local checkpoint
    print("Loading SAM 3 model...")
    processor = Sam3Processor.from_pretrained(str(CHECKPOINT))
    model = Sam3Model.from_pretrained(str(CHECKPOINT), torch_dtype=dtype).to(device)
    model.eval()

    # Load image
    image = Image.open(IMAGE_PATH).convert("RGB")
    w, h = image.size
    print(f"Image: {IMAGE_PATH.name} ({w}x{h})")

    # Run inference with text prompt
    print(f'Detecting "{PROMPT}"...')
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process: extract boxes and scores
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=[(h, w)],
        threshold=CONF_THRESHOLD,
    )[0]

    boxes = results["boxes"].cpu().tolist()
    scores = results["scores"].cpu().tolist()

    if not boxes:
        print("No persons detected.")
        return

    print(f"\nFound {len(boxes)} person(s):\n")
    print(f"{'#':<4} {'Confidence':<12} {'x1':>6} {'y1':>6} {'x2':>6} {'y2':>6}")
    print("-" * 44)
    for i, (box, score) in enumerate(zip(boxes, scores), start=1):
        x1, y1, x2, y2 = box
        print(f"{i:<4} {score:<12.3f} {x1:>6.1f} {y1:>6.1f} {x2:>6.1f} {y2:>6.1f}")


if __name__ == "__main__":
    main()
