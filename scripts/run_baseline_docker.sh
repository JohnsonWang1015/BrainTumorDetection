#!/usr/bin/env bash
set -euo pipefail

docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml run --rm trainer
