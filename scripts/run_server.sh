#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "venv not found at ${ROOT}/.venv; run: make install" >&2
  exit 1
fi

cd "${ROOT}"
mkdir -p logs
exec "${PYTHON}" -m uvicorn app.main:app \
  --host "${ETORACLE_HOST:-127.0.0.1}" \
  --port "${ETORACLE_PORT:-8787}"
