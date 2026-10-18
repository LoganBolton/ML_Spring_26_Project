# Usage: uv run python infer_toomers.py <image> [out_dir]
# Runs the best fine-tuned sweep model and the yolo26n baseline, saving annotated images, YOLO-format boxes, and counts.
import sys, json
from pathlib import Path
import cv2
from ultralytics import YOLO

img = Path(sys.argv[1])
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else img.parent
out_dir.mkdir(parents=True, exist_ok=True)
models = {
    "finetuned": ("sweeps/v10/trials/1776384144137_w0_0015/best.pt", 0.25),
    "baseline": ("yolo26n.pt", 0.10),
}
summary = {}
for name, (w, conf) in models.items():
    r = YOLO(w).predict(str(img), conf=conf, iou=0.7, imgsz=1280, device=0,
                        classes=[0], max_det=3000, verbose=False)[0]
    n = len(r.boxes)
    vis = r.orig_img.copy()
    for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy().astype(int):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    out = out_dir / f"{img.stem}_{name}_pred.jpg"
    cv2.imwrite(str(out), vis)
    txt = out_dir / f"{img.stem}_{name}_pred.txt"
    txt.unlink(missing_ok=True)  # save_txt appends, so clear any previous run
    r.save_txt(str(txt), save_conf=True)
    summary[name] = {"weights": w, "conf": conf, "count": n, "image": out.name}
    print(name, n, out)
(out_dir / f"{img.stem}_summary.json").write_text(json.dumps(summary, indent=2))
