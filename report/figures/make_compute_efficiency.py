"""Generate the compute-efficiency figure for report section 5.2.

Two panels:
  (a) inference latency per frame (log scale, bar chart)
  (b) accuracy-vs-compute Pareto: count MAE vs inference latency

Numbers come from the compare_models.ipynb run and Table 1 / Table 3 of the report.
Run:
    uv run python report/figures/make_compute_efficiency.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "compute_efficiency.png"

models = [
    {
        "name": "Pretrained\nYOLO26n",
        "mae": 3.10,
        "infer_ms": 14,
        "train_min": 0.0,
        "color": "tab:orange",
    },
    {
        "name": "SAM 3\nzero-shot",
        "mae": 5.00,
        "infer_ms": 500,
        "train_min": 0.0,
        "color": "tab:red",
    },
    {
        "name": "Fine-tuned\nYOLO26m",
        "mae": 1.27,
        "infer_ms": 32,
        "train_min": 10.0,
        "color": "tab:blue",
    },
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# --- Panel (a): inference latency bar chart, log scale ---
ax = axes[0]
names = [m["name"] for m in models]
infer = [m["infer_ms"] for m in models]
colors = [m["color"] for m in models]
bars = ax.bar(names, infer, color=colors, edgecolor="black", linewidth=0.6)
ax.set_yscale("log")
ax.set_ylabel("Inference latency (ms / frame, log scale)")
ax.set_title("(a) Per-frame inference cost")
ax.grid(alpha=0.3, axis="y", which="both")
for b, v in zip(bars, infer):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v} ms",
            ha="center", va="bottom", fontsize=9)
ax.set_ylim(8, 1200)

# --- Panel (b): accuracy-vs-compute Pareto ---
ax = axes[1]
for m in models:
    ax.scatter(m["infer_ms"], m["mae"], s=160, color=m["color"],
               edgecolor="black", linewidth=0.7, zorder=3)
    ax.annotate(m["name"].replace("\n", " "),
                (m["infer_ms"], m["mae"]),
                xytext=(8, 6), textcoords="offset points",
                fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("Inference latency (ms / frame, log scale)")
ax.set_ylabel("Count MAE (people / frame, lower is better)")
ax.set_title("(b) Accuracy vs. compute")
ax.grid(alpha=0.3, which="both")
ax.set_xlim(8, 1200)
ax.set_ylim(0, 6.0)

# Shade the Pareto-preferred corner (low latency + low error)
ax.axhspan(0, 2.0, xmin=0, xmax=0.5, alpha=0.08, color="tab:green")

fig.suptitle("Compute efficiency: fine-tuned YOLO26m is both faster than SAM 3 and more accurate than either baseline",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT}")
