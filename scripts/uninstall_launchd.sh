#!/usr/bin/env bash
set -euo pipefail

UID_NUM="$(id -u)"
SERVER_LABEL="ai.m1.eth-oracle"
DAILY_LABEL="ai.m1.eth-oracle-daily"
LAUNCH_AGENTS="${HOME}/Library/LaunchAgents"

launchctl bootout "gui/${UID_NUM}/${SERVER_LABEL}" 2>/dev/null || true
launchctl bootout "gui/${UID_NUM}/${DAILY_LABEL}" 2>/dev/null || true
rm -f "${LAUNCH_AGENTS}/${SERVER_LABEL}.plist" "${LAUNCH_AGENTS}/${DAILY_LABEL}.plist"
echo "removed ${SERVER_LABEL} and ${DAILY_LABEL}"
