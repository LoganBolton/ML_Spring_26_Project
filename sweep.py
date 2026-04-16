"""
Hyperparameter sweep for YOLO counting models.

Primary metric: count MSE on the validation set. Bounding-box tightness is
ignored — this is a counting task, so for each image we compare the number of
predicted boxes (after a confidence threshold) to the number of ground-truth
boxes, then average the squared error.

Each trial:
  1. Samples a config from the search space.
  2. Trains a YOLO model for --epochs epochs.
  3. Evaluates on the val split, sweeping confidence thresholds to find the
     threshold that minimises count MSE.
  4. Logs the result row to sweeps/<sweep-name>/results.csv and
     sweeps/<sweep-name>/results.jsonl.
  5. Keeps the top-K trial weight directories; deletes everything else.

Parallelism: one worker process per GPU. Each worker is pinned via
CUDA_VISIBLE_DEVICES and pulls independent random samples.

Usage:
    # Use both GPUs (default)
    uv run python sweep.py --sweep-name v2 --n-trials 500 --epochs 20

    # Single GPU
    uv run python sweep.py --sweep-name v1 --n-trials 200 --devices 0
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

# Headless matplotlib to avoid GUI backend hangs in spawned workers.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ---------------------------------------------------------------------------
# Search space. Keys are ultralytics train() kwargs unless noted.
# ---------------------------------------------------------------------------
SEARCH_SPACE: dict[str, list] = {
    # Model & size
    "model":        ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt"],
    "imgsz":        [960, 1280],
    "batch":        [8, 16],

    # Optimizer
    "optimizer":    ["SGD", "AdamW", "auto"],
    "lr0":          [1e-4, 5e-4, 1e-3, 3e-3, 1e-2],
    "lrf":          [0.005, 0.01, 0.05, 0.1],
    "momentum":     [0.85, 0.9, 0.937, 0.95],
    "weight_decay": [0.0, 1e-4, 5e-4, 1e-3],
    "warmup_epochs": [1.0, 3.0, 5.0],
    "cos_lr":       [False, True],

    # Loss weights — cls matters more than box for counting
    "box":          [2.0, 5.0, 7.5, 10.0],
    "cls":          [0.25, 0.5, 1.0, 2.0],
    "dfl":          [0.5, 1.5, 2.5],

    # Augmentation — colour
    "hsv_h":        [0.0, 0.015, 0.05],
    "hsv_s":        [0.3, 0.5, 0.7, 0.9],
    "hsv_v":        [0.2, 0.4, 0.6],

    # Augmentation — geometric
    "degrees":      [0.0, 5.0, 15.0],
    "translate":    [0.0, 0.1, 0.2],
    "scale":        [0.25, 0.5, 0.8],
    "fliplr":       [0.0, 0.5],
    "flipud":       [0.0, 0.1],

    # Augmentation — composite
    "mosaic":       [0.0, 0.5, 1.0],
    "mixup":        [0.0, 0.1, 0.25],
    "copy_paste":   [0.0, 0.2, 0.5],
    "erasing":      [0.0, 0.2, 0.4],
    "close_mosaic": [0, 5, 10, 20],

    # Regularization
    "dropout":      [0.0, 0.1, 0.2],
}

# Confidence thresholds to probe during evaluation (picks the best per trial)
CONF_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]

CSV_HEADER = [
    "trial_id", "worker_id", "gpu", "timestamp", "duration_sec", "status",
    "count_mse", "count_mae", "best_conf",
    "count_mse_true_mean", "count_mse_pred_mean",
    "n_val_images", "mAP50",
] + list(SEARCH_SPACE.keys()) + ["error"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sample_config(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def load_val_paths(dataset_yaml: Path) -> tuple[list[Path], list[int]]:
    """Parse dataset.yaml and return (val image paths, true counts)."""
    with open(dataset_yaml) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", dataset_yaml.parent))
    val_rel = cfg["val"]
    val_img_dir = root / val_rel
    # YOLO convention: images/... <-> labels/...
    val_lbl_dir = Path(str(val_img_dir).replace("/images/", "/labels/"))

    if not val_img_dir.is_dir():
        raise SystemExit(f"Val images dir not found: {val_img_dir}")
    if not val_lbl_dir.is_dir():
        raise SystemExit(f"Val labels dir not found: {val_lbl_dir}")

    img_paths: list[Path] = []
    true_counts: list[int] = []
    for p in sorted(val_img_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        lbl = val_lbl_dir / f"{p.stem}.txt"
        if lbl.exists():
            n = sum(1 for line in lbl.read_text().splitlines() if line.strip())
        else:
            n = 0
        img_paths.append(p)
        true_counts.append(n)
    return img_paths, true_counts


def evaluate_counting(
    model,
    img_paths: list[Path],
    true_counts: list[int],
    imgsz: int,
    device: str,
) -> dict:
    """Predict once per image with very low conf, filter by CONF_GRID in post."""
    true = np.array(true_counts, dtype=np.float64)
    per_conf_preds: dict[float, list[int]] = {c: [] for c in CONF_GRID}

    results_iter = model.predict(
        source=[str(p) for p in img_paths],
        conf=0.001,
        imgsz=imgsz,
        device=device,
        classes=[0],
        verbose=False,
        stream=True,
    )
    for res in results_iter:
        if res.boxes is None or len(res.boxes) == 0:
            confs = np.zeros(0)
        else:
            confs = res.boxes.conf.detach().cpu().numpy()
        for c in CONF_GRID:
            per_conf_preds[c].append(int((confs >= c).sum()))

    best = None
    for c in CONF_GRID:
        pred = np.array(per_conf_preds[c], dtype=np.float64)
        mse = float(((pred - true) ** 2).mean())
        mae = float(np.abs(pred - true).mean())
        entry = {
            "conf": c,
            "mse": mse,
            "mae": mae,
            "pred_mean": float(pred.mean()),
        }
        if best is None or mse < best["mse"]:
            best = entry

    return {
        "best_conf": best["conf"],
        "count_mse": best["mse"],
        "count_mae": best["mae"],
        "count_mse_pred_mean": best["pred_mean"],
        "count_mse_true_mean": float(true.mean()),
        "n_val_images": len(img_paths),
    }


def append_row(csv_path: Path, row: dict) -> None:
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def prune_trials(sweep_dir: Path, top_k: int) -> None:
    """Keep best.pt of the top-K 'ok' trials by count_mse; purge 'failed' trials
    and 'ok' trials that fell out of the top-K. In-progress trials (no CSV row
    yet) are left alone so concurrent workers' dirs aren't wiped mid-training.
    """
    csv_path = sweep_dir / "results.csv"
    if not csv_path.exists():
        return
    ok_rows: list[tuple[float, str]] = []
    failed_ids: set[str] = set()
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            tid = r.get("trial_id")
            if not tid:
                continue
            status = r.get("status")
            if status == "ok":
                try:
                    ok_rows.append((float(r["count_mse"]), tid))
                except Exception:
                    continue
            elif status == "failed":
                failed_ids.add(tid)
    ok_rows.sort()
    keep = {tid for _, tid in ok_rows[:top_k]}
    purge = failed_ids | {tid for _, tid in ok_rows[top_k:]}

    trials_dir = sweep_dir / "trials"
    if not trials_dir.is_dir():
        return
    for tdir in trials_dir.iterdir():
        if not tdir.is_dir():
            continue
        name = tdir.name
        if name in keep:
            # Keep only best.pt + small metadata, drop big artifacts
            for sub in tdir.rglob("*"):
                if sub.is_file() and sub.name not in {
                    "best.pt", "trial_config.json", "trial_result.json", "results.csv"
                }:
                    try:
                        sub.unlink()
                    except OSError:
                        pass
        elif name in purge:
            shutil.rmtree(tdir, ignore_errors=True)
        # else: not yet logged (in-progress) — leave alone.


@contextmanager
def timed():
    t0 = time.time()
    yield lambda: time.time() - t0


# ---------------------------------------------------------------------------
# Pre-download / pre-warm helpers (spawned as one-shot subprocesses)
# ---------------------------------------------------------------------------
def _download_model(name: str) -> None:
    # Hide GPUs — this is a pure download, no CUDA needed.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("MPLBACKEND", "Agg")
    from ultralytics import YOLO as _YOLO
    _YOLO(name)


def _warmup_dataset(model_name: str, data_yaml: str) -> None:
    """Force ultralytics to build labels.cache for train+val splits so concurrent
    workers don't race on the file on their first trial."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("MPLBACKEND", "Agg")
    from ultralytics import YOLO as _YOLO
    m = _YOLO(model_name)
    try:
        m.val(data=data_yaml, device="cpu", workers=0, verbose=False, plots=False)
    except Exception as e:
        print(f"[warmup] val() failed (non-fatal): {e}", flush=True)


# ---------------------------------------------------------------------------
# Worker (runs in its own process, one per GPU)
# ---------------------------------------------------------------------------
def worker_main(
    worker_id: int,
    gpu: int,
    n_trials: int,
    args_dict: dict,
    val_imgs: list[Path],
    true_counts: list[int],
    seed: int,
    csv_lock,
) -> None:
    # Unbuffered stdout/stderr so we can see where each worker is in real time.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    print(f"[worker {worker_id}] booting pid={os.getpid()} requested_gpu={gpu}", flush=True)

    # Pin this process to a single physical GPU BEFORE importing torch/ultralytics.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Stagger imports slightly so two workers don't race on ultralytics' settings.json
    # and HUB-file initialisation when both processes start at the exact same moment.
    time.sleep(worker_id * 1.5)

    print(f"[worker {worker_id}] importing ultralytics...", flush=True)
    import torch
    from ultralytics import YOLO

    # Verify pinning worked. With CUDA_VISIBLE_DEVICES=<gpu>, device_count()
    # must be 1 and current_device() must be 0.
    dev_count = torch.cuda.device_count()
    if dev_count != 1:
        print(f"[worker {worker_id}] WARNING: expected 1 visible CUDA device but "
              f"torch sees {dev_count}. CUDA_VISIBLE_DEVICES may not have taken "
              f"effect (env={os.environ.get('CUDA_VISIBLE_DEVICES')})", flush=True)
    try:
        torch.cuda.set_device(0)
    except Exception as e:
        print(f"[worker {worker_id}] torch.cuda.set_device(0) failed: {e}", flush=True)
    print(f"[worker {worker_id}] ultralytics ready | torch sees {dev_count} GPU(s) | "
          f"current={torch.cuda.current_device()}", flush=True)

    sweep_dir = Path(args_dict["sweep_dir"])
    data_yaml = Path(args_dict["data"]).resolve()
    csv_path = sweep_dir / "results.csv"
    jsonl_path = sweep_dir / "results.jsonl"

    rng = random.Random(seed)
    cache_val = args_dict["cache"] if args_dict["cache"].lower() != "false" else False

    print(f"[worker {worker_id}] pid={os.getpid()} gpu={gpu} seed={seed} trials={n_trials}",
          flush=True)

    for i in range(n_trials):
        trial_id = f"{int(time.time()*1000):013d}_w{worker_id}_{i:04d}"
        trial_dir = sweep_dir / "trials" / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        cfg = sample_config(rng)
        (trial_dir / "trial_config.json").write_text(json.dumps(cfg, indent=2))
        print(f"\n[w{worker_id} gpu{gpu}][trial {i+1}/{n_trials}] {trial_id}", flush=True)
        for k, v in cfg.items():
            print(f"    {k:14s} = {v}", flush=True)

        if args_dict["dry_run"]:
            continue

        row = {
            "trial_id": trial_id,
            "worker_id": worker_id,
            "gpu": gpu,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": 0.0,
            "status": "running",
            **cfg,
        }

        t_start = time.time()
        try:
            model = YOLO(cfg["model"])
            train_kwargs = {k: v for k, v in cfg.items() if k != "model"}
            # IMPORTANT: pass absolute path for `project` — ultralytics silently
            # prepends its default runs_dir ("runs/detect/") to any relative path,
            # which breaks our layout under sweeps/<name>/trials/<id>/run/.
            project_abs = str(trial_dir.resolve())
            model.train(
                data=str(data_yaml),
                epochs=args_dict["epochs"],
                patience=args_dict["patience"],
                # CUDA_VISIBLE_DEVICES has already restricted visibility, so
                # the only visible device for ultralytics is index 0.
                device=0,
                workers=args_dict["workers"],
                cache=cache_val,
                project=project_abs,
                name="run",
                exist_ok=True,
                verbose=False,
                plots=False,
                **train_kwargs,
            )

            run_dir = trial_dir / "run"
            weights_dir = run_dir / "weights"
            best_pt = weights_dir / "best.pt"
            last_pt = weights_dir / "last.pt"

            chosen_pt = None
            if best_pt.exists():
                chosen_pt = best_pt
            elif last_pt.exists():
                print(f"    [w{worker_id} warn] best.pt missing, falling back to last.pt",
                      flush=True)
                chosen_pt = last_pt
            else:
                present = []
                if run_dir.exists():
                    present = sorted(str(p.relative_to(run_dir))
                                     for p in run_dir.rglob("*") if p.is_file())
                raise FileNotFoundError(
                    f"no weights produced in {run_dir}. "
                    f"files present: {present[:30]}"
                )

            best_model = YOLO(str(chosen_pt))

            map50 = None
            try:
                metrics = best_model.val(
                    data=str(data_yaml),
                    imgsz=cfg["imgsz"],
                    device=0,
                    workers=args_dict["workers"],
                    verbose=False,
                    plots=False,
                )
                map50 = float(getattr(metrics.box, "map50", float("nan")))
            except Exception as e:
                print(f"    [w{worker_id} warn] val() failed: {e}", flush=True)

            eval_out = evaluate_counting(
                best_model, val_imgs, true_counts,
                imgsz=cfg["imgsz"], device=0,
            )

            shutil.copy2(chosen_pt, trial_dir / "best.pt")

            row.update(eval_out)
            row["mAP50"] = map50
            row["status"] = "ok"
            row["duration_sec"] = round(time.time() - t_start, 2)

            print(f"    [w{worker_id}] count_mse={row['count_mse']:.3f}  "
                  f"count_mae={row['count_mae']:.3f}  "
                  f"best_conf={row['best_conf']:.2f}  "
                  f"mAP50={row.get('mAP50')}  "
                  f"time={row['duration_sec']:.0f}s", flush=True)

        except Exception as e:
            row["status"] = "failed"
            row["error"] = f"{type(e).__name__}: {e}"
            row["duration_sec"] = round(time.time() - t_start, 2)
            print(f"    [w{worker_id} error] ({row['duration_sec']:.1f}s) {row['error']}",
                  flush=True)
            traceback.print_exc()

        with csv_lock:
            append_row(csv_path, row)
            append_jsonl(jsonl_path, row)
            (trial_dir / "trial_result.json").write_text(
                json.dumps(row, indent=2, default=str)
            )
            try:
                prune_trials(sweep_dir, top_k=args_dict["top_k"])
            except Exception as e:
                print(f"    [w{worker_id} warn] prune failed: {e}", flush=True)

    print(f"[worker {worker_id}] done", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="YOLO hyperparameter sweep for counting")
    p.add_argument("--data", type=str,
                   default="data/manual_label_4_15_26/dataset.yaml")
    p.add_argument("--sweep-name", type=str, default="v1")
    p.add_argument("--sweep-root", type=Path, default=Path("sweeps"))
    p.add_argument("--n-trials", type=int, default=200,
                   help="Total trials across ALL GPUs (split evenly).")
    p.add_argument("--epochs", type=int, default=30,
                   help="Epochs per trial. Lower = more trials, noisier metric.")
    p.add_argument("--devices", type=str, default="0,1",
                   help="Comma-separated CUDA device indices, one worker per device. "
                        "e.g. '0', '0,1', '0,1,2,3'. Use 'cpu' for CPU-only (1 worker).")
    p.add_argument("--seed", type=int, default=None,
                   help="Base seed; each worker offsets from this. Defaults to time.")
    p.add_argument("--top-k", type=int, default=20,
                   help="How many best-trial weight dirs to keep; others are pruned.")
    p.add_argument("--patience", type=int, default=15,
                   help="YOLO early-stopping patience per trial.")
    p.add_argument("--workers", type=int, default=4,
                   help="YOLO dataloader workers per trial (per GPU). Lower than "
                        "single-GPU default because two trials run concurrently.")
    p.add_argument("--cache", type=str, default="ram",
                   help="YOLO image cache: 'ram', 'disk', or 'false'.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print sampled configs without training.")
    args = p.parse_args()

    sweep_dir = args.sweep_root / args.sweep_name
    (sweep_dir / "trials").mkdir(parents=True, exist_ok=True)

    # Parse dataset once in the main process
    data_yaml = Path(args.data).resolve()
    val_imgs, true_counts = load_val_paths(data_yaml)
    print(f"[sweep] name={args.sweep_name}  devices={args.devices}  "
          f"total_trials={args.n_trials}  epochs={args.epochs}")
    print(f"[sweep] val: {len(val_imgs)} images, true count mean={np.mean(true_counts):.2f}, "
          f"sum={int(np.sum(true_counts))}")

    # Pre-download any missing model weights in a one-shot subprocess so the
    # main process never imports torch/ultralytics itself (prevents CUDA state
    # leaking into spawned workers and breaking CUDA_VISIBLE_DEVICES pinning).
    missing = [m for m in SEARCH_SPACE["model"] if not Path(m).exists()]
    ctx_pre = mp.get_context("spawn")
    if missing:
        print(f"[sweep] pre-downloading missing weights: {missing}")
        for m in missing:
            pre = ctx_pre.Process(target=_download_model, args=(m,))
            pre.start()
            pre.join()
            if pre.exitcode != 0:
                print(f"[sweep]   WARN pre-download of {m} failed (exit={pre.exitcode})")
            elif Path(m).exists():
                print(f"[sweep]   got {m}")

    # Pre-warm dataset labels.cache so concurrent workers don't race on it.
    warmup_model = next((m for m in SEARCH_SPACE["model"] if Path(m).exists()),
                        SEARCH_SPACE["model"][0])
    print(f"[sweep] warming dataset labels.cache with {warmup_model} ...")
    warm = ctx_pre.Process(target=_warmup_dataset,
                           args=(warmup_model, str(data_yaml)))
    warm.start()
    warm.join()
    print(f"[sweep]   warmup exit={warm.exitcode}")

    # Parse devices
    devices_raw = args.devices.strip()
    if devices_raw.lower() == "cpu":
        gpus = [-1]
    else:
        gpus = [int(x) for x in devices_raw.split(",") if x.strip() != ""]
    n_workers = len(gpus)

    # Split trials across workers (front-loaded if not divisible)
    base = args.n_trials // n_workers
    rem = args.n_trials % n_workers
    trials_per = [base + (1 if i < rem else 0) for i in range(n_workers)]

    base_seed = args.seed if args.seed is not None else int(time.time())

    args_dict = {
        "sweep_dir": str(sweep_dir),
        "data": args.data,
        "epochs": args.epochs,
        "patience": args.patience,
        "workers": args.workers,
        "cache": args.cache,
        "top_k": args.top_k,
        "dry_run": args.dry_run,
    }

    ctx = mp.get_context("spawn")
    csv_lock = ctx.Lock()
    procs: list[mp.Process] = []
    for wid, gpu in enumerate(gpus):
        seed = (base_seed + wid * 1_000_003) & 0xFFFFFFFF
        proc = ctx.Process(
            target=worker_main,
            args=(wid, gpu, trials_per[wid], args_dict,
                  val_imgs, true_counts, seed, csv_lock),
            name=f"sweep-worker-{wid}-gpu{gpu}",
        )
        proc.start()
        procs.append(proc)

    try:
        for proc in procs:
            proc.join()
    except KeyboardInterrupt:
        print("\n[sweep] KeyboardInterrupt — terminating workers")
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=10)

    print(f"\n[sweep] done. Results: {sweep_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
