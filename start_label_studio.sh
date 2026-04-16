#!/usr/bin/env bash
# Start Label Studio with local-files serving enabled.
# Usage: ./start_label_studio.sh [port]

set -euo pipefail

cd "$(dirname "$0")"

PORT="${1:-8080}"

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/
export CSRF_TRUSTED_ORIGINS=https://mixture-analysts-href-discounts.trycloudflare.com

exec uv run label-studio start --port "$PORT" --no-browser
