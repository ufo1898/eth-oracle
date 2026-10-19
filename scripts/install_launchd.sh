#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
UID_NUM="$(id -u)"
LAUNCH_AGENTS="${HOME}/Library/LaunchAgents"
SERVER_LABEL="ai.m1.eth-oracle"
DAILY_LABEL="ai.m1.eth-oracle-daily"
SERVER_PLIST="${LAUNCH_AGENTS}/${SERVER_LABEL}.plist"
DAILY_PLIST="${LAUNCH_AGENTS}/${DAILY_LABEL}.plist"

if [[ ! -x "${PYTHON}" ]]; then
  echo "venv not found at ${ROOT}/.venv; run: make install" >&2
  exit 1
fi

mkdir -p "${LAUNCH_AGENTS}" "${ROOT}/logs" "${ROOT}/reports"

cat >"${SERVER_PLIST}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${SERVER_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${PYTHON}</string>
    <string>-m</string>
    <string>uvicorn</string>
    <string>app.main:app</string>
    <string>--host</string>
    <string>127.0.0.1</string>
    <string>--port</string>
    <string>8787</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${ROOT}</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${ROOT}/logs/uvicorn.out.log</string>
  <key>StandardErrorPath</key>
  <string>${ROOT}/logs/uvicorn.err.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
    <key>NO_PROXY</key>
    <string>*</string>
  </dict>
</dict>
</plist>
EOF

cat >"${DAILY_PLIST}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${DAILY_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${PYTHON}</string>
    <string>-m</string>
    <string>app.daily</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${ROOT}</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>8</integer>
    <key>Minute</key>
    <integer>10</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>${ROOT}/logs/daily.launchd.out.log</string>
  <key>StandardErrorPath</key>
  <string>${ROOT}/logs/daily.launchd.err.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>
</dict>
</plist>
EOF

launchctl bootout "gui/${UID_NUM}/${SERVER_LABEL}" 2>/dev/null || true
launchctl bootout "gui/${UID_NUM}/${DAILY_LABEL}" 2>/dev/null || true
launchctl bootstrap "gui/${UID_NUM}" "${SERVER_PLIST}"
launchctl bootstrap "gui/${UID_NUM}" "${DAILY_PLIST}"
launchctl enable "gui/${UID_NUM}/${SERVER_LABEL}"
launchctl enable "gui/${UID_NUM}/${DAILY_LABEL}"

echo "installed:"
echo "  server: ${SERVER_LABEL} (http://127.0.0.1:8787)"
echo "  daily:  ${DAILY_LABEL} (08:10 local, logs/daily.launchd.err.log)"
