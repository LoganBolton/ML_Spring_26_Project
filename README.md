# ML Project

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Review Labels

```bash
uv run python sam3/review.py --images-dir data/toomer_test --labels-dir data/toomer_test/raw
```

Opens Label Studio at `http://localhost:8080` with images and YOLO bounding boxes pre-loaded for review. Login: `admin@localhost` / `admin1234`.

## Run

```bash
uv run python yolo.py
```
