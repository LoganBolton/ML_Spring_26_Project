# moondream_env

Isolated uv-managed environment for moondream fine-tuning scripts.

The `moondream` package requires `pillow<11.0.0`, which conflicts with
`label-studio-sdk`'s `pillow>=11.3.0` requirement in the main project, so it
lives in its own venv here.

## Usage

From the repository root:

```bash
# Install / sync deps
uv sync --project moondream_env

# Run a script that uses moondream (cwd stays at repo root so relative paths work)
uv run --project moondream_env python train_moondream.py
```
