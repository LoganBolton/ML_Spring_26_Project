"""
Launch Label Studio with SAM3-generated YOLO pre-annotations for human review.

Usage:
    uv run python sam3/review.py
    uv run python sam3/review.py --images-dir data/raw_images --labels-dir data/labels/raw --port 8080
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

LABEL_CONFIG = """<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="#FF0000"/>
  </RectangleLabels>
</View>"""

PROJECT_TITLE = "SAM3 Annotation Review"


def parse_yolo_label(label_path: Path) -> list[dict]:
    """Parse a YOLO label file into Label Studio prediction results.

    YOLO format: class_id x_center y_center width height (all normalized 0-1)
    Label Studio format: x, y, width, height (all as percentages 0-100)
    where x, y is the top-left corner.
    """
    results = []
    if not label_path.exists():
        return results

    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        _, xc, yc, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        # Convert YOLO center-based normalized coords to LS top-left percentage coords
        x_pct = (xc - w / 2) * 100
        y_pct = (yc - h / 2) * 100
        w_pct = w * 100
        h_pct = h * 100

        results.append({
            "value": {
                "x": x_pct,
                "y": y_pct,
                "width": w_pct,
                "height": h_pct,
                "rotation": 0,
                "rectanglelabels": ["person"],
            },
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
        })

    return results


def wait_for_label_studio(base_url: str, timeout: int = 60) -> bool:
    """Poll Label Studio until it responds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/api/version", timeout=3)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


def ensure_label_studio_running(port: int) -> None:
    """Start Label Studio if it isn't already running on the given port."""
    base_url = f"http://localhost:{port}"
    try:
        r = requests.get(f"{base_url}/api/version", timeout=3)
        if r.status_code == 200:
            print(f"Label Studio already running at {base_url}")
            return
    except requests.ConnectionError:
        pass

    print(f"Starting Label Studio on port {port}...")
    env = os.environ.copy()
    # Allow Label Studio to serve files from the local filesystem
    env.setdefault("LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED", "true")
    env.setdefault("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT", "/")

    subprocess.Popen(
        [sys.executable, "-m", "label_studio", "start", "--port", str(port), "--no-browser"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    if not wait_for_label_studio(base_url):
        raise SystemExit("Label Studio failed to start within 60 seconds.")
    print(f"Label Studio started at {base_url}")


def get_session(base_url: str) -> requests.Session:
    """Return an authenticated requests.Session using cookie-based auth.

    Label Studio 1.23+ disabled legacy token auth, so we use session cookies.
    """
    email = os.environ.get("LABEL_STUDIO_EMAIL", "admin@localhost")
    password = os.environ.get("LABEL_STUDIO_PASSWORD", "admin1234")
    session = requests.Session()

    # Try signup first (works on a fresh instance with no users)
    session.get(f"{base_url}/user/signup", timeout=10)
    csrf = session.cookies.get("csrftoken", "")
    r = session.post(
        f"{base_url}/user/signup",
        data={"email": email, "password": password, "csrfmiddlewaretoken": csrf},
        headers={"Referer": f"{base_url}/user/signup"},
        timeout=10,
    )

    if r.url.rstrip("/").endswith("/user/signup"):
        # User already exists — log in instead
        session = requests.Session()
        session.get(f"{base_url}/user/login", timeout=10)
        csrf = session.cookies.get("csrftoken", "")
        session.post(
            f"{base_url}/user/login",
            data={"email": email, "password": password, "csrfmiddlewaretoken": csrf},
            headers={"Referer": f"{base_url}/user/login"},
            timeout=10,
        )

    # Verify we are authenticated
    r = session.get(f"{base_url}/api/current-user/whoami", timeout=5)
    if not r.ok:
        raise SystemExit(
            "Could not authenticate with Label Studio.\n"
            "Set LABEL_STUDIO_EMAIL / LABEL_STUDIO_PASSWORD in your environment,\n"
            "or log in at the Label Studio UI first."
        )
    print(f"Authenticated as {r.json().get('email', '?')}")
    return session


def _api(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    """Make an API call with the session's CSRF token."""
    csrf = session.cookies.get("csrftoken", "")
    headers = kwargs.pop("headers", {})
    headers.setdefault("X-CSRFToken", csrf)
    headers.setdefault("Content-Type", "application/json")
    return session.request(method, url, headers=headers, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Label Studio for annotation review")
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw_images"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/labels/raw"))
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    images_dir: Path = args.images_dir.resolve()
    labels_dir: Path = args.labels_dir.resolve()
    port: int = args.port
    base_url = f"http://localhost:{port}"

    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No images found in {images_dir}")

    # 1. Ensure Label Studio is running
    ensure_label_studio_running(port)

    # 2. Authenticate via session cookies
    session = get_session(base_url)

    # 3. Create project (or reuse existing one with the same title)
    r = _api(session, "GET", f"{base_url}/api/projects/")
    r.raise_for_status()
    projects = r.json().get("results", [])

    project_id = None
    for p in projects:
        if p["title"] == PROJECT_TITLE:
            project_id = p["id"]
            print(f"Reusing existing project: {PROJECT_TITLE} (id={project_id})")
            break

    if project_id is None:
        r = _api(session, "POST", f"{base_url}/api/projects/",
                 data=json.dumps({"title": PROJECT_TITLE, "label_config": LABEL_CONFIG}))
        r.raise_for_status()
        project_id = r.json()["id"]
        print(f"Created project: {PROJECT_TITLE} (id={project_id})")

    # 4. Set up local file storage
    _api(session, "POST", f"{base_url}/api/storages/localfiles",
         data=json.dumps({
             "project": project_id,
             "title": "Raw Images",
             "path": str(images_dir),
             "use_blob_urls": True,
         }))

    # 5. Build tasks with pre-annotations
    tasks = []
    for img_path in images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        predictions = parse_yolo_label(label_path)

        task: dict = {"data": {"image": f"/data/local-files/?d={img_path}"}}
        if predictions:
            task["predictions"] = [{"result": predictions, "model_version": "sam3"}]
        tasks.append(task)

    # 6. Import tasks
    r = _api(session, "POST", f"{base_url}/api/projects/{project_id}/import",
             data=json.dumps(tasks))
    r.raise_for_status()
    print(f"Imported {len(tasks)} task(s)")

    # 7. Print URL
    project_url = f"{base_url}/projects/{project_id}"
    print(f"\nOpen in your browser:\n  {project_url}\n")
    print("After reviewing, export corrected labels in YOLO format from the Label Studio UI.")


if __name__ == "__main__":
    main()
