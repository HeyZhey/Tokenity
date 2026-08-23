#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT/apps/TokenityControl"

CONFIGURATION="${TOKENITY_BUILD_CONFIGURATION:-debug}"
BUNDLE_IDENTIFIER="${TOKENITY_BUNDLE_IDENTIFIER:-ai.tokenity.control.stable}"
BUNDLE_NAME="${TOKENITY_BUNDLE_NAME:-Tokenity}"
BUNDLE_VERSION="${TOKENITY_BUNDLE_VERSION:-1}"
BUNDLE_SHORT_VERSION="${TOKENITY_BUNDLE_SHORT_VERSION:-0.1.0}"

cd "$APP_DIR"
swift build -c "$CONFIGURATION" >&2

BIN_DIR="$(swift build -c "$CONFIGURATION" --show-bin-path)"
EXECUTABLE="$BIN_DIR/TokenityControl"
APP_BUNDLE="${TOKENITY_APP_BUNDLE_PATH:-$BIN_DIR/Tokenity.app}"
ICON_SOURCE="$APP_DIR/Resources/AppIcon.icns"
BRAND_RESOURCE_DIR="$APP_DIR/Sources/TokenityControl/Resources"

if [[ ! -x "$EXECUTABLE" ]]; then
  echo "TokenityControl executable was not produced by swift build." >&2
  exit 1
fi

rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources"
cp "$EXECUTABLE" "$APP_BUNDLE/Contents/MacOS/TokenityControl"

if [[ -f "$ICON_SOURCE" ]]; then
  cp "$ICON_SOURCE" "$APP_BUNDLE/Contents/Resources/AppIcon.icns"
fi

for resource_name in \
  TokenityBrandLockup.png \
  TokenityBrandLockupDark.png \
  TokenityBrandMark.png \
  TokenityBrandMarkDark.png; do
  resource_source="$BRAND_RESOURCE_DIR/$resource_name"
  if [[ ! -f "$resource_source" ]]; then
    echo "Required brand resource is missing: $resource_source" >&2
    exit 1
  fi
  cp "$resource_source" "$APP_BUNDLE/Contents/Resources/$resource_name"
done

cat > "$APP_BUNDLE/Contents/Info.plist" <<PLIST
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
  <string>${BUNDLE_IDENTIFIER}</string>
  <key>CFBundleIconFile</key>
  <string>AppIcon</string>
  <key>CFBundleName</key>
  <string>${BUNDLE_NAME}</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>${BUNDLE_SHORT_VERSION}</string>
  <key>CFBundleVersion</key>
  <string>${BUNDLE_VERSION}</string>
  <key>LSMinimumSystemVersion</key>
  <string>26.2</string>
  <key>NSHighResolutionCapable</key>
  <true/>
  <key>NSLocalNetworkUsageDescription</key>
  <string>Tokenity connects to Node Agents on your local network to coordinate distributed inference.</string>
  <key>NSPrincipalClass</key>
  <string>NSApplication</string>
</dict>
</plist>
PLIST

plutil -lint "$APP_BUNDLE/Contents/Info.plist" >/dev/null

if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - "$APP_BUNDLE" >/dev/null 2>&1 || true
fi

echo "$APP_BUNDLE"
