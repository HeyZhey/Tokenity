#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT/apps/TokenityControl"

# Keep App Bubble unambiguous while the old prototype still exists on disk.
pkill -f "/Users/zxc/Documents/MLX-Distributed/apps/TokenityControl" 2>/dev/null || true
pkill -f "$APP_DIR/.build" 2>/dev/null || true

cd "$APP_DIR"
swift build

EXECUTABLE="$APP_DIR/.build/arm64-apple-macosx/debug/TokenityControl"
APP_BUNDLE="$APP_DIR/.build/arm64-apple-macosx/debug/TokenityControl.app"

if [[ ! -x "$EXECUTABLE" ]]; then
  echo "TokenityControl executable was not produced by swift build." >&2
  exit 1
fi

rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
cp "$EXECUTABLE" "$APP_BUNDLE/Contents/MacOS/TokenityControl"
cat > "$APP_BUNDLE/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>TokenityControl</string>
  <key>CFBundleIdentifier</key>
  <string>ai.tokenity.control.dev</string>
  <key>CFBundleName</key>
  <string>TokenityControl</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>14.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
  <key>NSPrincipalClass</key>
  <string>NSApplication</string>
</dict>
</plist>
PLIST

open -F -n "$APP_BUNDLE"
/usr/bin/osascript -e 'tell application id "ai.tokenity.control.dev" to activate' >/dev/null 2>&1 || true
echo "Opened $APP_BUNDLE"
