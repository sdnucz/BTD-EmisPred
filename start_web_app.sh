#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$(dirname "$0")"
exec "${PYTHON_BIN}" web_app/app.py --host "${HOST}" --port "${PORT}"
