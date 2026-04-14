from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO("yolo26n.pt")

# Perform object detection on the test image (persons only)
results = model("data/test_toomers.png", classes=[0])
results[0].show()

# Print detection results
for box in results[0].boxes:
    conf = float(box.conf[0])
    print(f"  person: {conf:.2f}")
