"""
Back up Label Studio annotations for the review project.

Exports both JSON (Label Studio native) and YOLO formats to a timestamped
directory so they can be restored or versioned.

Usage:
    uv run python sam3/backup.py
    uv run python sam3/backup.py --output-dir data/backups --port 8080
"""

import argparse
import json
import os
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import requests

PROJECT_TITLE = "SAM3 Annotation Review"


def get_session(base_url: str) -> requests.Session:
    email = os.environ.get("LABEL_STUDIO_EMAIL", "admin@localhost")
    password = os.environ.get("LABEL_STUDIO_PASSWORD", "admin1234")
    session = requests.Session()
    session.get(f"{base_url}/user/login", timeout=10)
    csrf = session.cookies.get("csrftoken", "")
    session.post(
        f"{base_url}/user/login",
        data={"email": email, "password": password, "csrfmiddlewaretoken": csrf},
        headers={"Referer": f"{base_url}/user/login"},
        timeout=10,
    )
    r = session.get(f"{base_url}/api/current-user/whoami", timeout=5)
    if not r.ok:
        raise SystemExit("Could not authenticate with Label Studio.")
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description="Back up Label Studio annotations")
    parser.add_argument("--output-dir", type=Path, default=Path("data/backups"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--project-title", default=PROJECT_TITLE)
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"
    session = get_session(base_url)

    # Find project by title
    r = session.get(f"{base_url}/api/projects/", timeout=10)
    r.raise_for_status()
    project = next(
        (p for p in r.json().get("results", []) if p["title"] == args.project_title),
        None,
    )
    if project is None:
        raise SystemExit(f"Project not found: {args.project_title}")

    project_id = project["id"]
    annotated = project.get("num_tasks_with_annotations", "?")
    print(f"Project: {args.project_title} (id={project_id}) — {annotated} annotated tasks")

    # Create timestamped backup directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = args.output_dir / f"{ts}_project{project_id}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export JSON (includes full Label Studio task + annotation data)
    r = session.get(
        f"{base_url}/api/projects/{project_id}/export",
        params={"exportType": "JSON"},
        timeout=60,
    )
    r.raise_for_status()
    (backup_dir / "annotations.json").write_bytes(r.content)
    tasks = json.loads(r.content)
    print(f"  Saved annotations.json ({len(tasks)} tasks)")

    # 2. Export YOLO (zip with labels/ and classes.txt)
    r = session.get(
        f"{base_url}/api/projects/{project_id}/export",
        params={"exportType": "YOLO"},
        timeout=60,
    )
    if r.ok and r.headers.get("content-type", "").startswith("application/zip"):
        yolo_dir = backup_dir / "yolo"
        yolo_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(BytesIO(r.content)) as z:
            z.extractall(yolo_dir)
        label_count = len(list((yolo_dir / "labels").glob("*.txt"))) if (yolo_dir / "labels").exists() else 0
        print(f"  Saved yolo/ ({label_count} label files)")
    else:
        print(f"  YOLO export skipped (status {r.status_code})")

    print(f"\nBackup written to: {backup_dir}")


if __name__ == "__main__":
    main()
