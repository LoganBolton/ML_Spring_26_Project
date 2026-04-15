"""
Launch Label Studio with SAM3-generated YOLO pre-annotations for human review.

Usage:
    uv run python sam3/review.py
    uv run python sam3/review.py --images-dir data/raw_images --labels-dir data/labels/raw --port 8080
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from label_studio_sdk import LabelStudio

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


def get_api_key(base_url: str) -> str:
    """Return an API key, using the env var or creating a default user."""
    key = os.environ.get("LABEL_STUDIO_API_KEY")
    if key:
        return key

    # Try to sign up a default user (works on first run of a fresh instance)
    signup_url = f"{base_url}/user/signup"
    payload = {
        "email": "admin@localhost",
        "password": "admin1234",
    }
    try:
        r = requests.post(signup_url, json=payload, timeout=10)
        if r.status_code == 201:
            token = r.json().get("token")
            if token:
                print("Created default admin user (admin@localhost / admin1234)")
                return token
    except Exception:
        pass

    # Try to log in with the default credentials
    login_url = f"{base_url}/api/auth/login"
    try:
        session = requests.Session()
        session.get(base_url, timeout=5)
        csrf = session.cookies.get("csrftoken", "")
        r = session.post(
            login_url,
            json={"email": "admin@localhost", "password": "admin1234"},
            headers={"X-CSRFToken": csrf},
            timeout=10,
        )
        if r.ok:
            # Fetch token from the API
            token_r = session.get(f"{base_url}/api/current-user/token", timeout=5)
            if token_r.ok:
                return token_r.json().get("token", "")
    except Exception:
        pass

    raise SystemExit(
        "Could not obtain a Label Studio API key.\n"
        "Set LABEL_STUDIO_API_KEY in your environment, or start Label Studio\n"
        "manually and create an account first."
    )


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

    # 2. Get API key
    api_key = get_api_key(base_url)
    client = LabelStudio(base_url=base_url, api_key=api_key)

    # 3. Create project (or reuse existing one with the same title)
    existing = client.projects.list()
    project = None
    for p in existing:
        if p.title == PROJECT_TITLE:
            project = p
            print(f"Reusing existing project: {PROJECT_TITLE} (id={project.id})")
            break

    if project is None:
        project = client.projects.create(
            title=PROJECT_TITLE,
            label_config=LABEL_CONFIG,
        )
        print(f"Created project: {PROJECT_TITLE} (id={project.id})")

    # 4. Set up local file storage
    client.import_storage.local.create(
        project=project.id,
        title="Raw Images",
        path=str(images_dir),
        use_blob_urls=True,
    )

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
    response = client.projects.import_tasks(id=project.id, request=tasks)
    print(f"Imported {len(tasks)} task(s)")

    # 7. Print URL
    project_url = f"{base_url}/projects/{project.id}"
    print(f"\nOpen in your browser:\n  {project_url}\n")
    print("After reviewing, export corrected labels in YOLO format from the Label Studio UI.")


if __name__ == "__main__":
    main()
