#!/usr/bin/env bash
set -euo pipefail

uv sync --frozen

echo "Environment ready via uv sync."
