from PIL import Image, ImageDraw
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load moondream2 from HuggingFace. Ampere (RTX 3090) can't run the FP8
# `md.vl(local=True)` build, but the HF checkpoint ships in bf16/fp16.
MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-06-21"  # pin for reproducibility; bump as needed

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda"},
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

# Load an image
image = Image.open("data/stream5/all/frame_00000_0h00m00s_12-01-33pm.jpg")

# Generate a caption
caption = model.caption(image, length="normal")["caption"]
print("Caption:", caption)

# Ask a question
answer = model.query(image, "What's in this image?")["answer"]
print("Answer:", answer)

# Detect persons and draw bounding boxes
OBJECT = "person"
OUTPUT_PATH = "moondream_detections.png"

detections = model.detect(image, OBJECT)["objects"]
print(f"Detected {len(detections)} {OBJECT}(s)")

annotated = image.convert("RGB").copy()
draw = ImageDraw.Draw(annotated)
W, H = annotated.size

for det in detections:
    # moondream2 returns normalized [0, 1] coords
    x0, y0 = det["x_min"] * W, det["y_min"] * H
    x1, y1 = det["x_max"] * W, det["y_max"] * H
    draw.rectangle([x0, y0, x1, y1], outline="red", width=1)

annotated.save(OUTPUT_PATH)
print(f"Saved annotated image to {OUTPUT_PATH}")
