#!/usr/bin/env bash
# Launch the Brain Tumor Detection web interface
#
# Usage:
#   ./scripts/launch_app.sh            # default port 7860
#   ./scripts/launch_app.sh --port 8080 # custom port
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

# Try uv first, fall back to direct python
if command -v uv &>/dev/null && [ -f ".venv/bin/python" ]; then
    exec uv run python -m idh_glioma.app "$@"
else
    PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}" \
        exec python -m idh_glioma.app "$@"
fi
