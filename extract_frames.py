"""
extract_frames.py
-----------------
Extracts every frame from 1fps trimmed videos, stamps each frame with its
timestamp, and logs full metadata (JSON + CSV) for the entire run.

HOW TO RUN
----------
Default (looks for videos in a folder called "Trim Videos" next to this script):
    uv run python extract_frames.py

Point it at any folder on your machine using --input-dir:
    uv run python extract_frames.py --input-dir "/path/to/your/video/folder"

All options:
    uv run python extract_frames.py \
        --input-dir  "/path/to/videos"   # folder containing your .mp4/.mov/etc. files
        --output-dir "/path/to/output"   # where frames + metadata are written (created if missing)
        --quality    95                  # JPEG quality 1-100

OUTPUT LAYOUT
-------------
    <output-dir>/
        <video_stem>/
            frame_00000_0h00m00s.jpg
            frame_00001_0h00m01s.jpg
            ...
        metadata.json   <- full run + per-frame metadata
        metadata.csv    <- same data in flat tabular form (one row per frame)

DEPENDENCIES
------------
Python packages (handled automatically by uv — just run `uv sync`):
    opencv-python, Pillow

System tool required: ffprobe (bundled with ffmpeg)
    Mac:     brew install ffmpeg
    Windows: winget install ffmpeg
             OR download from https://ffmpeg.org/download.html and add to PATH
    Linux:   sudo apt install ffmpeg       # Debian/Ubuntu
             sudo dnf install ffmpeg       # Fedora
             sudo pacman -S ffmpeg         # Arch

SETUP (any machine)
-------------------
1. Install ffmpeg using the command for your OS above
2. Install uv if you don't have it:
       Mac/Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh
       Windows:    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
3. Run `uv sync` in this folder to install Python dependencies
4. Run the script (see HOW TO RUN above)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2  # OpenCV — used for reading video frames and drawing the timestamp overlay

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# File extensions recognised as videos when scanning the input directory
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

# Timestamp overlay appearance
FONT             = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE       = 0.7
FONT_THICKNESS   = 2
TEXT_COLOR       = (255, 255, 255)   # white text
BG_COLOR         = (0,   0,   0)    # black background rectangle behind text
PADDING          = 8                 # pixels from the top-left corner


# ---------------------------------------------------------------------------
# Helper: time formatting
# ---------------------------------------------------------------------------

def format_hms_filename(seconds: float) -> str:
    """
    Convert elapsed seconds to a compact string safe for use in filenames.
    Example: 3661 -> '1h01m01s'
    """
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h}h{m:02d}m{s:02d}s"


def format_hms_display(seconds: float) -> str:
    """
    Convert elapsed seconds to a human-readable H:MM:SS string for the overlay.
    Example: 3661 -> '1:01:01'
    """
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Helper: video metadata via ffprobe
# ---------------------------------------------------------------------------

def probe_video(video_path: Path) -> dict:
    """
    Run ffprobe on a video file and return its full stream + format info as a dict.
    ffprobe is part of ffmpeg.  Install on Mac with: brew install ffmpeg
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",                # suppress noisy ffprobe logs
        "-print_format", "json",      # output as JSON
        "-show_streams",              # include codec/resolution/fps info
        "-show_format",               # include duration/bit-rate/container info
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path.name}:\n{result.stderr}")
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Helper: draw timestamp onto a frame
# ---------------------------------------------------------------------------

def stamp_frame(frame_bgr, timestamp_str: str):
    """
    Draw a timestamp string in the top-left corner of a BGR frame.
    A filled black rectangle is drawn behind the text for readability.
    Returns a new frame (does not modify the original in-place).
    """
    img = frame_bgr.copy()

    # Measure how much space the text will occupy
    (text_w, text_h), baseline = cv2.getTextSize(timestamp_str, FONT, FONT_SCALE, FONT_THICKNESS)

    # Top-left anchor of the text
    x = PADDING
    y = text_h + PADDING

    # Draw black background rectangle so text is visible on any frame colour
    cv2.rectangle(
        img,
        (x - 4, y - text_h - 4),          # top-left corner of rectangle
        (x + text_w + 4, y + baseline + 2), # bottom-right corner
        BG_COLOR, -1                        # -1 = filled
    )

    # Draw white text on top of the rectangle
    cv2.putText(img, timestamp_str, (x, y),
                FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# Core: extract all frames from one video
# ---------------------------------------------------------------------------

def process_video(video_path: Path, output_dir: Path, jpeg_quality: int = 95) -> dict:
    """
    Opens a video, reads every frame, stamps the timestamp, saves as JPEG.

    Parameters
    ----------
    video_path   : path to the source video file
    output_dir   : root output directory; frames go into a subfolder named after the video
    jpeg_quality : JPEG compression quality (1 = smallest file, 100 = best quality)

    Returns
    -------
    A dict containing all metadata collected for this video (used later for JSON/CSV logging).
    """
    video_stem = video_path.stem                     # filename without extension
    frame_dir  = output_dir / video_stem             # e.g. frames_output/stream5_1fps_v2/
    frame_dir.mkdir(parents=True, exist_ok=True)

    # ---- Probe the video to collect format metadata before opening it ----
    probe     = probe_video(video_path)
    vid_stream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
        {}
    )
    fmt = probe.get("format", {})

    # Parse frames-per-second from the "r_frame_rate" field (stored as "num/den")
    fps_str  = vid_stream.get("r_frame_rate", "1/1")
    num, den = (int(x) for x in fps_str.split("/"))
    fps      = num / den if den else 1.0

    duration_sec        = float(fmt.get("duration", vid_stream.get("duration", 0)))
    total_frames_probed = int(vid_stream.get("nb_frames", 0)) or int(duration_sec * fps)

    # ---- Open the video with OpenCV ----
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path.name}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    process_start  = time.time()
    frame_records  = []   # will hold one dict per extracted frame
    frame_idx      = 0

    print(f"\n[{video_stem}]  {width}x{height}  {fps:.3f} fps  ~{total_frames_probed} frames")

    # ---- Main extraction loop ----
    while True:
        ret, frame = cap.read()   # ret=False means end of video
        if not ret:
            break

        # For a 1fps video, frame index directly equals elapsed seconds
        elapsed_sec       = frame_idx / fps
        timestamp_display = format_hms_display(elapsed_sec)   # for the overlay
        timestamp_label   = format_hms_filename(elapsed_sec)  # for the filename

        # Stamp the timestamp onto the frame
        stamped = stamp_frame(frame, timestamp_display)

        # Build the output filename, e.g. frame_00042_0h00m42s.jpg
        out_filename = f"frame_{frame_idx:05d}_{timestamp_label}.jpg"
        out_path     = frame_dir / out_filename

        # Save the stamped frame as a JPEG
        cv2.imwrite(str(out_path), stamped, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        # Record per-frame metadata
        frame_records.append({
            "frame_index":     frame_idx,
            "elapsed_seconds": round(elapsed_sec, 3),
            "timestamp":       timestamp_display,
            "filename":        out_filename,
            "file_size_bytes": out_path.stat().st_size,
        })

        # Print a progress update every 100 frames so the user can see it's running
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx:>6}  t={timestamp_display}", flush=True)

        frame_idx += 1

    cap.release()   # always release the video capture when done
    process_elapsed = time.time() - process_start

    # ---- Assemble the full metadata dict for this video ----
    video_meta = {
        # Source file info
        "video_file":               video_path.name,
        "video_path":               str(video_path.resolve()),
        "file_size_bytes":          video_path.stat().st_size,
        "format_name":              fmt.get("format_name"),
        "codec":                    vid_stream.get("codec_name"),
        "bit_rate_bps":             int(fmt.get("bit_rate", 0) or 0),
        # Video properties
        "width":                    width,
        "height":                   height,
        "fps":                      fps,
        "duration_seconds":         duration_sec,
        "total_frames_probed":      total_frames_probed,
        # Extraction results
        "output_directory":         str(frame_dir.resolve()),
        "total_frames_extracted":   frame_idx,
        "jpeg_quality":             jpeg_quality,
        # Timing
        "process_start_utc":        datetime.utcfromtimestamp(process_start).isoformat() + "Z",
        "process_duration_sec":     round(process_elapsed, 2),
        # Per-frame records (included in JSON only; flattened in CSV)
        "frames":                   frame_records,
    }

    print(f"  Done: {frame_idx} frames saved in {process_elapsed:.1f}s")
    return video_meta


# ---------------------------------------------------------------------------
# Metadata writers
# ---------------------------------------------------------------------------

def write_json(metadata: dict, output_dir: Path):
    """Write the full run metadata (including nested per-frame records) to metadata.json."""
    path = output_dir / "metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata JSON → {path}")


def write_csv(metadata: dict, output_dir: Path):
    """
    Write a flat CSV where every row represents one frame.
    Video-level fields are repeated on each row so the CSV is self-contained.
    """
    path = output_dir / "metadata.csv"
    rows = []

    for video_meta in metadata["videos"]:
        # Separate the per-frame list from the video-level fields
        base_fields = {k: v for k, v in video_meta.items() if k != "frames"}
        for frame in video_meta["frames"]:
            rows.append({**base_fields, **frame})   # merge video fields + frame fields

    if not rows:
        print("No rows to write to CSV.")
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Metadata CSV  → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames from 1fps videos, stamp timestamps, and log metadata.\n\n"
            "The --input-dir argument lets you point this script at any folder on your "
            "machine — it does NOT need to be called 'Trim Videos'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default="Trim Videos",
        help=(
            "Path to the folder containing your video files. "
            "Can be any name/location (default: 'Trim Videos' next to this script)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="frames_output",
        help="Path where extracted frames and metadata will be written. Created if it doesn't exist. (default: frames_output)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality for saved frames, 1 (smallest) to 100 (best). (default: 95)",
    )
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Validate that the input directory exists before doing anything
    if not input_dir.exists():
        print(
            f"ERROR: input directory not found: {input_dir}\n"
            f"Use --input-dir to specify the correct path.\n"
            f"Example: uv run python extract_frames.py --input-dir \"/path/to/my/videos\"",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create the output directory if it doesn't already exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all video files in the input directory (non-recursive)
    videos = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        print(f"No video files found in '{input_dir}'. Supported formats: {VIDEO_EXTENSIONS}")
        sys.exit(0)

    print(f"Found {len(videos)} video(s) in '{input_dir}'")
    for v in videos:
        print(f"  {v.name}")

    # Top-level metadata for the entire run
    run_start    = time.time()
    run_metadata = {
        "run_start_utc":    datetime.utcnow().isoformat() + "Z",
        "input_directory":  str(input_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "jpeg_quality":     args.quality,
        "videos":           [],
    }

    # Process each video in turn
    for video_path in videos:
        try:
            video_meta = process_video(video_path, output_dir, jpeg_quality=args.quality)
            run_metadata["videos"].append(video_meta)
        except Exception as e:
            print(f"ERROR processing {video_path.name}: {e}", file=sys.stderr)

    # Summarise the full run
    run_metadata["run_duration_sec"]       = round(time.time() - run_start, 2)
    run_metadata["total_frames_extracted"] = sum(
        v["total_frames_extracted"] for v in run_metadata["videos"]
    )

    # Write both metadata files
    write_json(run_metadata, output_dir)
    write_csv(run_metadata, output_dir)

    print(
        f"\nAll done.  "
        f"{run_metadata['total_frames_extracted']} total frames extracted "
        f"in {run_metadata['run_duration_sec']}s."
    )


if __name__ == "__main__":
    main()
