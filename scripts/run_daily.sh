#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "venv not found at ${ROOT}/.venv; run: make install" >&2
  exit 1
fi

cd "${ROOT}"
mkdir -p logs reports
LOG="logs/daily-$(date +%Y-%m-%d).log"
exec "${PYTHON}" -m app.daily "$@" >>"${LOG}" 2>&1
