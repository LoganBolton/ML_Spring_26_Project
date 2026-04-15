"""
extract_frames.py
-----------------
Extracts every frame from 1fps trimmed videos, names each file with its
elapsed timestamp AND real-world wall-clock time, and logs full metadata
(JSON + CSV) for the entire run.

No timestamp is burned into the image itself — it lives only in the filename.

HOW TO RUN
----------
Basic (looks for videos in "Trim Videos" next to this script):
    uv run python extract_frames.py

With a log file so filenames include real-world time of day:
    uv run python extract_frames.py \
        --log-file stream_record.log \
        --stream-name stream5

Point it at any folder on your machine:
    uv run python extract_frames.py --input-dir "/path/to/your/video/folder"

All options:
    uv run python extract_frames.py \
        --input-dir   "Trim Videos"           # folder with video files
        --output-dir  "Trim Videos/Video Extraction/Stream 5"  # where to write output
        --log-file    stream_record.log        # log file with recording start times
        --stream-name stream5                  # which stream to look up in the log
        --quality     95                       # JPEG quality 1-100

FILENAME FORMAT
---------------
Without --log-file:
    frame_00042_0h00m42s.jpg

With --log-file and --stream-name:
    frame_00042_0h00m42s_12-02-15pm.jpg
    (wall-clock time = recording start time + elapsed seconds)

OUTPUT LAYOUT
-------------
    <output-dir>/
        <video_stem>/
            frame_00000_0h00m00s_12-01-33pm.jpg
            frame_00001_0h00m01s_12-01-34pm.jpg
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
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2  # OpenCV — used for reading video frames

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# File extensions recognised as videos when scanning the input directory
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

# Log file timestamp format: "2026-03-27 12:01:33,344"
LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Helper: time formatting
# ---------------------------------------------------------------------------

def format_hms_filename(seconds: float) -> str:
    """
    Convert elapsed seconds to a compact string safe for filenames.
    Example: 3661 -> '1h01m01s'
    """
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h}h{m:02d}m{s:02d}s"


def format_wallclock_filename(dt: datetime) -> str:
    """
    Format a datetime as a filesystem-safe time-of-day string.
    Uses dashes instead of colons so it works on Windows too.
    Example: 2026-03-27 14:05:09 -> '02-05-09pm'
    """
    return dt.strftime("%I-%M-%S%p").lower()   # e.g. '02-05-09pm'


# ---------------------------------------------------------------------------
# Helper: parse recording start time from log file
# ---------------------------------------------------------------------------

def parse_stream_start_time(log_file: Path, stream_name: str) -> datetime:
    """
    Scan the log file for the block that mentions <stream_name>.mp4 and
    return the datetime when that recording started.

    The log format looks like:
        2026-03-27 12:01:33,344 [INFO] === Stream recorder started ===
        2026-03-27 12:01:33,344 [INFO] Duration: 72 hours | Output: E:\\...\\stream5.mp4

    We find the Output line that matches the stream name, then read the
    timestamp from that same line (both lines share the same second).
    """
    log_text = log_file.read_text(encoding="utf-8", errors="replace")
    lines    = log_text.splitlines()

    # Pattern: a line whose Output field contains the stream name (case-insensitive)
    target = stream_name.lower()
    for line in lines:
        if "output:" in line.lower() and target in line.lower():
            # Extract the leading timestamp, e.g. "2026-03-27 12:01:33,344"
            match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if match:
                return datetime.strptime(match.group(1), LOG_TIMESTAMP_FORMAT)

    raise ValueError(
        f"Could not find a start time for '{stream_name}' in {log_file}.\n"
        f"Make sure --stream-name matches the filename in the log "
        f"(e.g. 'stream5' matches 'stream5.mp4')."
    )


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
        "-v", "quiet",             # suppress noisy ffprobe logs
        "-print_format", "json",   # output as JSON
        "-show_streams",           # codec / resolution / fps
        "-show_format",            # duration / bit-rate / container
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path.name}:\n{result.stderr}")
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Core: extract all frames from one video
# ---------------------------------------------------------------------------

def process_video(
    video_path: Path,
    output_dir: Path,
    jpeg_quality: int = 95,
    recording_start: datetime = None,
) -> dict:
    """
    Opens a video, reads every frame, saves as JPEG.
    Filename encodes elapsed time and (optionally) real-world wall-clock time.
    No timestamp is drawn onto the image itself.

    Parameters
    ----------
    video_path       : path to the source video file
    output_dir       : root output dir; frames go into a subfolder named after the video
    jpeg_quality     : JPEG compression quality (1 = smallest, 100 = best)
    recording_start  : real-world datetime when the recording began (from log file).
                       If provided, wall-clock time is appended to each filename.
    """
    video_stem = video_path.stem                  # filename without extension
    frame_dir  = output_dir / video_stem          # e.g. .../Stream 5/stream5_1fps_v2/
    frame_dir.mkdir(parents=True, exist_ok=True)

    # ---- Probe the video for format metadata ----
    probe      = probe_video(video_path)
    vid_stream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
        {}
    )
    fmt = probe.get("format", {})

    # Parse fps from "r_frame_rate" which is stored as "num/den"
    fps_str  = vid_stream.get("r_frame_rate", "1/1")
    num, den = (int(x) for x in fps_str.split("/"))
    fps      = num / den if den else 1.0

    duration_sec        = float(fmt.get("duration", vid_stream.get("duration", 0)))
    total_frames_probed = int(vid_stream.get("nb_frames", 0)) or int(duration_sec * fps)

    # ---- Open with OpenCV ----
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path.name}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    process_start = time.time()
    frame_records = []   # one dict per extracted frame (for metadata logging)
    frame_idx     = 0

    print(f"\n[{video_stem}]  {width}x{height}  {fps:.3f} fps  ~{total_frames_probed} frames")
    if recording_start:
        print(f"  Recording started at: {recording_start.strftime('%Y-%m-%d %I:%M:%S %p')}")

    # ---- Main extraction loop ----
    while True:
        ret, frame = cap.read()   # ret=False signals end of video
        if not ret:
            break

        # For a 1fps video, frame index == elapsed seconds from start of recording
        elapsed_sec    = frame_idx / fps
        elapsed_label  = format_hms_filename(elapsed_sec)   # e.g. '0h00m42s'

        # Build filename — append wall-clock time if a recording start was provided
        if recording_start:
            wall_time    = recording_start + timedelta(seconds=elapsed_sec)
            clock_label  = format_wallclock_filename(wall_time)   # e.g. '12-02-15pm'
            out_filename = f"frame_{frame_idx:05d}_{elapsed_label}_{clock_label}.jpg"
            wall_time_str = wall_time.strftime("%Y-%m-%d %I:%M:%S %p")
        else:
            out_filename  = f"frame_{frame_idx:05d}_{elapsed_label}.jpg"
            wall_time_str = None

        out_path = frame_dir / out_filename

        # Save the raw frame — no text burned into the image
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        # Record per-frame metadata for JSON/CSV output
        frame_record = {
            "frame_index":     frame_idx,
            "elapsed_seconds": round(elapsed_sec, 3),
            "elapsed_label":   elapsed_label,
            "filename":        out_filename,
            "file_size_bytes": out_path.stat().st_size,
        }
        if wall_time_str:
            frame_record["wall_clock_time"] = wall_time_str

        frame_records.append(frame_record)

        # Progress update every 100 frames
        if frame_idx % 100 == 0:
            msg = f"  frame {frame_idx:>6}  elapsed={elapsed_label}"
            if recording_start:
                msg += f"  wall={wall_time.strftime('%I:%M:%S %p')}"
            print(msg, flush=True)

        frame_idx += 1

    cap.release()   # always release the capture when done
    process_elapsed = time.time() - process_start

    # ---- Assemble full metadata dict for this video ----
    video_meta = {
        # Source file
        "video_file":             video_path.name,
        "video_path":             str(video_path.resolve()),
        "file_size_bytes":        video_path.stat().st_size,
        "format_name":            fmt.get("format_name"),
        "codec":                  vid_stream.get("codec_name"),
        "bit_rate_bps":           int(fmt.get("bit_rate", 0) or 0),
        # Video properties
        "width":                  width,
        "height":                 height,
        "fps":                    fps,
        "duration_seconds":       duration_sec,
        "total_frames_probed":    total_frames_probed,
        # Recording start (from log file, if provided)
        "recording_start":        recording_start.isoformat() if recording_start else None,
        # Extraction results
        "output_directory":       str(frame_dir.resolve()),
        "total_frames_extracted": frame_idx,
        "jpeg_quality":           jpeg_quality,
        # Timing
        "process_start_utc":      datetime.utcfromtimestamp(process_start).isoformat() + "Z",
        "process_duration_sec":   round(process_elapsed, 2),
        # Per-frame records (nested in JSON, flattened in CSV)
        "frames":                 frame_records,
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
    Write a flat CSV where every row is one frame.
    Video-level fields are repeated on each row so the file is self-contained.
    """
    path = output_dir / "metadata.csv"
    rows = []

    for video_meta in metadata["videos"]:
        base_fields = {k: v for k, v in video_meta.items() if k != "frames"}
        for frame in video_meta["frames"]:
            rows.append({**base_fields, **frame})

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
            "Extract frames from 1fps videos and name them with timestamps.\n\n"
            "Use --log-file and --stream-name to include the real-world wall-clock\n"
            "time of day in each filename (e.g. frame_00042_0h00m42s_12-02-15pm.jpg).\n\n"
            "The --input-dir argument accepts any folder path — it does NOT need to\n"
            "be called 'Trim Videos'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default="Trim Videos",
        help="Folder containing your video files. (default: 'Trim Videos')",
    )
    parser.add_argument(
        "--output-dir",
        default="frames_output",
        help="Where to write extracted frames and metadata. Created if missing. (default: frames_output)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to the stream_record.log file. Used to look up recording start times.",
    )
    parser.add_argument(
        "--stream-name",
        default=None,
        help="Stream name to look up in the log file (e.g. 'stream5'). Required if --log-file is set.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality 1 (smallest) to 100 (best). (default: 95)",
    )
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Validate input directory
    if not input_dir.exists():
        print(
            f"ERROR: input directory not found: {input_dir}\n"
            f"Use --input-dir to specify the correct path.\n"
            f"Example: uv run python extract_frames.py --input-dir \"/path/to/my/videos\"",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse recording start time from log file if provided
    recording_start = None
    if args.log_file:
        if not args.stream_name:
            print("ERROR: --stream-name is required when --log-file is provided.", file=sys.stderr)
            sys.exit(1)
        log_path = Path(args.log_file)
        if not log_path.exists():
            print(f"ERROR: log file not found: {log_path}", file=sys.stderr)
            sys.exit(1)
        try:
            recording_start = parse_stream_start_time(log_path, args.stream_name)
            print(f"Found recording start time for '{args.stream_name}': {recording_start}")
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect video files, filtered by stream name if provided
    videos = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if args.stream_name:
        videos = [p for p in videos if args.stream_name.lower() in p.stem.lower()]
    if not videos:
        print(f"No video files found in '{input_dir}'. Supported formats: {VIDEO_EXTENSIONS}")
        sys.exit(0)

    print(f"Found {len(videos)} video(s) in '{input_dir}'")
    for v in videos:
        print(f"  {v.name}")

    # Top-level run metadata
    run_start    = time.time()
    run_metadata = {
        "run_start_utc":    datetime.utcnow().isoformat() + "Z",
        "input_directory":  str(input_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "jpeg_quality":     args.quality,
        "log_file":         str(Path(args.log_file).resolve()) if args.log_file else None,
        "stream_name":      args.stream_name,
        "recording_start":  recording_start.isoformat() if recording_start else None,
        "videos":           [],
    }

    # Process each video
    for video_path in videos:
        try:
            video_meta = process_video(
                video_path,
                output_dir,
                jpeg_quality=args.quality,
                recording_start=recording_start,
            )
            run_metadata["videos"].append(video_meta)
        except Exception as e:
            print(f"ERROR processing {video_path.name}: {e}", file=sys.stderr)

    run_metadata["run_duration_sec"]       = round(time.time() - run_start, 2)
    run_metadata["total_frames_extracted"] = sum(
        v["total_frames_extracted"] for v in run_metadata["videos"]
    )

    write_json(run_metadata, output_dir)
    write_csv(run_metadata, output_dir)

    print(
        f"\nAll done.  "
        f"{run_metadata['total_frames_extracted']} total frames extracted "
        f"in {run_metadata['run_duration_sec']}s."
    )


if __name__ == "__main__":
    main()
