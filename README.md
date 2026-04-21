# Pedestrian Counting at Toomer's Corner

Fine-tuned YOLO26m for person counting on the Toomer's Corner public livestream at Auburn University. Achieves count MAE of 1.27 vs 3.10 for a pretrained YOLO26n baseline and 5.00 for SAM 3 zero-shot.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## Scripts

- `sam3/annotate.py` — generate SAM 3 pre-annotations (YOLO format)
- `sam3/review.py` — launch Label Studio for human annotation review
- `sam3/backup.py` — export Label Studio annotations to disk
- `prepare_dataset.py` — build train/val split and `dataset.yaml`
- `train.py` — single fine-tuning run
- `sweep.py` — multi-GPU random hyperparameter search
- `yolo.py` — quick inference demo
- `compare_models.ipynb` — side-by-side model comparison and plots
- `monitor_training.ipynb` — training curves and validation evaluation

## Reproducing Results

```bash
# 1. Annotate frames with SAM 3
uv run python sam3/annotate.py --input-dir data/raw_images --output-dir data/labels/raw

# 2. Review and correct labels in Label Studio
uv run python sam3/review.py --images-dir data/raw_images --labels-dir data/labels/raw

# 3. Build train/val split
uv run python prepare_dataset.py --dataset-dir data/manual_label_4_16_26_v2 --split 0.8 --seed 42

# 4. Run hyperparameter sweep
uv run python sweep.py --sweep-name v1 --n-trials 200 --epochs 30 --devices 0,1
```

Results are logged to `sweeps/<name>/results.csv`. Open `compare_models.ipynb` to evaluate and compare all models.
