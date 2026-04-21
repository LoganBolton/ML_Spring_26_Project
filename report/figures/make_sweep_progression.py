"""Generate the hyperparameter-sweep progression figure for report section 4.2.

Reads sweeps/<round>/results.csv (eight rounds: v1, v3, v4, v5, v7, v8, v9, v10)
and renders two panels:
  (a) per-round distribution of validation count MSE (every successful trial
      as a dot, with the per-round best highlighted).
  (b) running best-so-far count MSE across all 195 successful trials, ordered
      chronologically, showing monotonic improvement from search-space pruning.

Run:
    uv run python report/figures/make_sweep_progression.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "sweep_progression.png"

ROUNDS = ["v1", "v3", "v4", "v5", "v7", "v8", "v9", "v10"]

# model -> colour
MODEL_COLORS = {
    "yolo26n.pt": "tab:gray",
    "yolo26s.pt": "tab:orange",
    "yolo26m.pt": "tab:blue",
}


def load_round(name: str) -> list[dict]:
    """Return all successful trials from sweeps/<name>/results.csv."""
    path = REPO / "sweeps" / name / "results.csv"
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            try:
                rows.append({
                    "round": name,
                    "timestamp": r["timestamp"],
                    "count_mse": float(r["count_mse"]),
                    "model": r["model"],
                })
            except (KeyError, ValueError):
                continue
    return rows


all_trials: list[dict] = []
for r in ROUNDS:
    all_trials.extend(load_round(r))

print(f"Loaded {len(all_trials)} successful trials across {len(ROUNDS)} rounds.")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

# --- Panel (a): per-round distribution ---
ax = axes[0]
for i, rnd in enumerate(ROUNDS):
    xs, ys, cs = [], [], []
    for t in all_trials:
        if t["round"] != rnd:
            continue
        xs.append(i + np.random.RandomState(hash(t["timestamp"]) & 0xFFFF).uniform(-0.18, 0.18))
        ys.append(t["count_mse"])
        cs.append(MODEL_COLORS.get(t["model"], "black"))
    if ys:
        ax.scatter(xs, ys, c=cs, s=28, alpha=0.75, edgecolor="white", linewidth=0.5, zorder=2)

# per-round best line
best_per_round = []
for rnd in ROUNDS:
    vals = [t["count_mse"] for t in all_trials if t["round"] == rnd]
    best_per_round.append(min(vals) if vals else np.nan)
valid_idx = [i for i, v in enumerate(best_per_round) if not np.isnan(v)]
ax.plot([i for i in valid_idx],
        [best_per_round[i] for i in valid_idx],
        color="black", linewidth=1.5, marker="D", markersize=7,
        markerfacecolor="white", markeredgewidth=1.5, zorder=3,
        label="per-round best")

ax.set_xticks(range(len(ROUNDS)))
ax.set_xticklabels(ROUNDS)
ax.set_xlabel("Sweep round")
ax.set_ylabel("Validation count MSE (lower is better)")
ax.set_title("(a) Count MSE per trial, by sweep round")
ax.grid(alpha=0.3, axis="y")

# Legend for model colours + per-round-best marker
handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS["yolo26n.pt"],
               markersize=8, label="YOLO26n", markeredgecolor="white"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS["yolo26s.pt"],
               markersize=8, label="YOLO26s", markeredgecolor="white"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS["yolo26m.pt"],
               markersize=8, label="YOLO26m", markeredgecolor="white"),
    plt.Line2D([0], [0], color="black", linewidth=1.5, marker="D",
               markersize=7, markerfacecolor="white", markeredgewidth=1.5,
               label="per-round best"),
]
ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

# --- Panel (b): running best-so-far ---
ax = axes[1]
ordered = sorted(all_trials, key=lambda t: t["timestamp"])
idx = np.arange(1, len(ordered) + 1)
running = np.minimum.accumulate([t["count_mse"] for t in ordered])

# raw trials as small dots
ax.scatter(idx, [t["count_mse"] for t in ordered],
           c=[MODEL_COLORS.get(t["model"], "black") for t in ordered],
           s=12, alpha=0.45, edgecolor="none", zorder=2)

# running best line
ax.plot(idx, running, color="tab:red", linewidth=2.2, label="running best", zorder=3)

# annotate final best
final_best = running[-1]
ax.annotate(f"best: MSE = {final_best:.2f}",
            xy=(idx[-1], final_best), xytext=(-120, 40),
            textcoords="offset points", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

# also mark pretrained-baseline MSE for reference
BASELINE_MSE = 15.01
ax.axhline(BASELINE_MSE, linestyle="--", color="tab:gray", linewidth=1)
ax.text(idx[-1], BASELINE_MSE + 0.4,
        f"pretrained YOLO26n baseline = {BASELINE_MSE:.2f}",
        ha="right", va="bottom", fontsize=8, color="tab:gray")

ax.set_xlabel("Trial index (chronological, 195 total)")
ax.set_ylabel("Validation count MSE (lower is better)")
ax.set_title("(b) Running best-so-far across all trials")
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

fig.suptitle(
    "Hyperparameter sweep: 195 trials over 8 rounds drive count MSE from 7.06 → 2.79",
    fontsize=11, y=1.02,
)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT}")
