from PIL import Image
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
