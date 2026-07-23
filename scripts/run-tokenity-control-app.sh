#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT/apps/TokenityControl"
BUNDLE_IDENTIFIER="${TOKENITY_BUNDLE_IDENTIFIER:-ai.tokenity.control.stable}"

# Keep App Bubble unambiguous while the old prototype still exists on disk.
pkill -f "/Users/zxc/Documents/MLX-Distributed/apps/TokenityControl" 2>/dev/null || true
pkill -f "$APP_DIR/.build" 2>/dev/null || true

APP_BUNDLE="$("$ROOT/scripts/build-tokenity-control-app.sh")"

open -F -n "$APP_BUNDLE"
/usr/bin/osascript -e "tell application id \"${BUNDLE_IDENTIFIER}\" to activate" >/dev/null 2>&1 || true
echo "Opened $APP_BUNDLE"
