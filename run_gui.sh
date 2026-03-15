#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$SCRIPT_DIR/dist/FAST-Calib-GUI-linux"

if [[ ! -x "$APP" ]]; then
  echo "Packaged GUI not found: $APP" >&2
  exit 1
fi

exec "$APP"
