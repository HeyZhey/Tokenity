#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${TOKENITY_VERSION:-0.1.0}"
DIST_DIR="$ROOT/dist"
WORK_DIR="$DIST_DIR/package-work"
PAYLOAD_DIR="$WORK_DIR/payload"
SCRIPTS_DIR="$WORK_DIR/scripts"
DMG_ROOT="$WORK_DIR/dmg-root"
RUNTIME_CACHE="$DIST_DIR/runtime-cache/TokenityRuntime"
RUNTIME_SOURCE="${TOKENITY_RUNTIME_SOURCE:-/Users/Shared/TokenityRuntime}"
NODE_AGENT_PACKAGE_MODE="${TOKENITY_NODE_AGENT_PACKAGE:-auto}"
PKG_PATH="$DIST_DIR/Tokenity-NodeAgent-Runtime-${VERSION}.pkg"
DMG_PATH="$DIST_DIR/Tokenity-${VERSION}.dmg"
DMG_CHECKSUM_PATH="$DMG_PATH.sha256"
APP_BUNDLE="$DMG_ROOT/TokenityControl.app"
BACKEND_SOURCE="$ROOT/tokenity/serving/distributed_openai.py"

if ! grep -q '^class TokenityDistributedRuntime' "$BACKEND_SOURCE"; then
  echo "Refusing to package a skeleton-only Tokenity backend." >&2
  exit 1
fi

rsync_excludes=(
  --exclude ".DS_Store"
  --exclude ".git"
  --exclude ".venv"
  --exclude ".pytest_cache"
  --exclude ".swiftpm"
  --exclude ".build"
  --exclude "DerivedData"
  --exclude "__pycache__"
  --exclude "*.pyc"
  --exclude "dist"
)

runtime_excludes=(
  --exclude ".DS_Store"
  --exclude ".lock"
  --exclude "CACHEDIR.TAG"
  --exclude ".git"
  --exclude "cache"
)

copy_dir() {
  local source="$1"
  local destination="$2"
  mkdir -p "$destination"
  rsync -a --delete "${rsync_excludes[@]}" "$source" "$destination/"
}

copy_runtime_dir() {
  local source="$1"
  local destination="$2"
  mkdir -p "$destination"
  rsync -a --delete --delete-excluded "${runtime_excludes[@]}" "$source" "$destination/"
}

case "$NODE_AGENT_PACKAGE_MODE" in
  auto)
    if [[ -d "$RUNTIME_SOURCE/current/.venv" || -d "$RUNTIME_CACHE/current/.venv" ]]; then
      BUILD_NODE_AGENT_PACKAGE=1
    else
      BUILD_NODE_AGENT_PACKAGE=0
    fi
    ;;
  required)
    BUILD_NODE_AGENT_PACKAGE=1
    ;;
  skip)
    BUILD_NODE_AGENT_PACKAGE=0
    ;;
  *)
    echo "TOKENITY_NODE_AGENT_PACKAGE must be auto, required, or skip." >&2
    exit 1
    ;;
esac

if [[ "${TOKENITY_INCLUDE_MODEL:-0}" == "1" && "$BUILD_NODE_AGENT_PACKAGE" != "1" ]]; then
  echo "TOKENITY_INCLUDE_MODEL=1 requires TOKENITY_NODE_AGENT_PACKAGE=required and a runtime source." >&2
  exit 1
fi

echo "Preparing package workspace..."
rm -rf "$WORK_DIR"
rm -f "$PKG_PATH" "$DMG_PATH" "$DMG_CHECKSUM_PATH"
mkdir -p "$SCRIPTS_DIR" "$DMG_ROOT" "$DIST_DIR"

echo "Building TokenityControl.app..."
TOKENITY_BUILD_CONFIGURATION=release \
TOKENITY_APP_BUNDLE_PATH="$APP_BUNDLE" \
TOKENITY_BUNDLE_IDENTIFIER=ai.tokenity.control \
"$ROOT/scripts/build-tokenity-control-app.sh" >/dev/null

codesign --verify --deep --strict "$APP_BUNDLE"
ln -s /Applications "$DMG_ROOT/Applications"

if [[ "$BUILD_NODE_AGENT_PACKAGE" == "1" ]]; then
mkdir -p "$PAYLOAD_DIR/Users/Shared/TokenityCode" \
  "$PAYLOAD_DIR/Users/Shared/TokenityRuntime" \
  "$PAYLOAD_DIR/Users/Shared/TokenityModels" \
  "$PAYLOAD_DIR/Library/LaunchDaemons"

echo "Copying Tokenity backend code..."
copy_dir "$ROOT/" "$PAYLOAD_DIR/Users/Shared/TokenityCode"

echo "Preparing Tokenity runtime..."
if [[ -d "$RUNTIME_SOURCE/current/.venv" ]]; then
  copy_runtime_dir "$RUNTIME_SOURCE/" "$RUNTIME_CACHE"
elif [[ -d "$RUNTIME_CACHE/current/.venv" ]]; then
  echo "Using the cached Tokenity runtime at $RUNTIME_CACHE."
else
  echo "Tokenity runtime not found. Set TOKENITY_RUNTIME_SOURCE to a local directory." >&2
  exit 1
fi
copy_runtime_dir "$RUNTIME_CACHE/" "$PAYLOAD_DIR/Users/Shared/TokenityRuntime"

if [[ "${TOKENITY_INCLUDE_MODEL:-0}" == "1" ]]; then
  MODEL_NAME="${TOKENITY_MODEL_NAME:-Qwen3.5-122B-A10B-4bit}"
  MODEL_SOURCE="${TOKENITY_MODEL_SOURCE:-/Users/Shared/TokenityModels/${MODEL_NAME}}"
  MODEL_DEST="$PAYLOAD_DIR/Users/Shared/TokenityModels/$MODEL_NAME"
  echo "Copying model payload. This can produce a very large DMG..."
  mkdir -p "$MODEL_DEST"
  if [[ -e "$MODEL_SOURCE" ]]; then
    rsync -aL --delete --exclude ".DS_Store" "$MODEL_SOURCE/" "$MODEL_DEST/"
  else
    echo "Model not found. Set TOKENITY_MODEL_SOURCE to a local directory." >&2
    exit 1
  fi
else
  echo "Skipping model weights. Set TOKENITY_INCLUDE_MODEL=1 to build a large offline model DMG."
fi

cat > "$PAYLOAD_DIR/Library/LaunchDaemons/ai.tokenity.node-agent.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ai.tokenity.node-agent</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/Shared/TokenityRuntime/current/.venv/bin/python</string>
    <string>-u</string>
    <string>-m</string>
    <string>tokenity</string>
    <string>node-agent</string>
    <string>--host</string>
    <string>0.0.0.0</string>
    <string>--port</string>
    <string>9100</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/Users/Shared/TokenityRuntime/current/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>PYTHONPATH</key>
    <string>/Users/Shared/TokenityCode</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ProcessType</key>
  <string>Interactive</string>
  <key>StandardOutPath</key>
  <string>/Users/Shared/TokenityLogs/node-agent.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/Shared/TokenityLogs/node-agent.err</string>
</dict>
</plist>
PLIST

cat > "$SCRIPTS_DIR/postinstall" <<'SCRIPT'
#!/bin/zsh
set -eu

NODE_AGENT_PLIST="/Library/LaunchDaemons/ai.tokenity.node-agent.plist"
TB_PLIST="/Library/LaunchDaemons/ai.tokenity.thunderbolt-keepalive.plist"
TB_SCRIPT="/usr/local/bin/tokenity-tb-keepalive"
PYTHON="/Users/Shared/TokenityRuntime/current/.venv/bin/python"
AGENT_USER="$(/usr/bin/stat -f '%Su' /dev/console 2>/dev/null || true)"

if [[ -n "$AGENT_USER" && "$AGENT_USER" != "root" && "$AGENT_USER" != "loginwindow" ]]; then
  /usr/libexec/PlistBuddy -c "Add :UserName string $AGENT_USER" "$NODE_AGENT_PLIST" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c "Set :UserName $AGENT_USER" "$NODE_AGENT_PLIST"
fi

/bin/mkdir -p /Users/Shared/TokenityLogs /Users/Shared/TokenityModels /usr/local/bin
/usr/sbin/chown root:wheel "$NODE_AGENT_PLIST" 2>/dev/null || true
/usr/sbin/chown -R "$AGENT_USER":staff /Users/Shared/TokenityLogs 2>/dev/null || true
/bin/chmod 644 "$NODE_AGENT_PLIST"
/bin/chmod -R a+rX /Users/Shared/TokenityCode /Users/Shared/TokenityRuntime /Users/Shared/TokenityModels 2>/dev/null || true

cat > "$TB_SCRIPT" <<'KEEPALIVE'
#!/bin/zsh
iface="$1"
local_ip="$2"
peer_ip="$3"
failures=0
reset_link() {
  /bin/date "+%Y-%m-%d %H:%M:%S resetting $iface ($1)" >> /Users/Shared/TokenityLogs/thunderbolt-keepalive.log
  /sbin/ifconfig "$iface" down
  /bin/sleep 1
  /sbin/ifconfig "$iface" inet "$local_ip" netmask 255.255.255.252 up
  /bin/sleep 3
  failures=0
}
while true; do
  if ! /sbin/ifconfig "$iface" | /usr/bin/grep -q "inet $local_ip"; then
    /sbin/ifconfig "$iface" inet "$local_ip" netmask 255.255.255.252 up
  fi
  if ! /sbin/ifconfig "$iface" | /usr/bin/grep -q "status: active"; then
    reset_link inactive
    /bin/sleep 2
    continue
  fi
  if /sbin/ping -c 1 -W 1000 "$peer_ip" >/dev/null 2>&1; then
    failures=0
  else
    failures=$((failures + 1))
  fi
  if [ "$failures" -ge 3 ]; then
    reset_link ping-fail
  fi
  /bin/sleep 2
done
KEEPALIVE
/bin/chmod 755 "$TB_SCRIPT"

configure_thunderbolt_keepalive() {
  local iface="$1"
  local local_ip="$2"
  local peer_ip="$3"

  /sbin/ifconfig "$iface" inet "$local_ip" netmask 255.255.255.252 up 2>/dev/null || true
  cat > "$TB_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ai.tokenity.thunderbolt-keepalive</string>
  <key>ProgramArguments</key>
  <array>
    <string>$TB_SCRIPT</string>
    <string>$iface</string>
    <string>$local_ip</string>
    <string>$peer_ip</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/Users/Shared/TokenityLogs/thunderbolt-keepalive.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/Shared/TokenityLogs/thunderbolt-keepalive.err</string>
</dict>
</plist>
PLIST
  /usr/sbin/chown root:wheel "$TB_PLIST"
  /bin/chmod 644 "$TB_PLIST"
  /bin/launchctl bootout system "$TB_PLIST" 2>/dev/null || true
  /bin/launchctl bootstrap system "$TB_PLIST" 2>/dev/null || true
  /bin/launchctl enable system/ai.tokenity.thunderbolt-keepalive 2>/dev/null || true
}

lan_ips="$(/sbin/ifconfig | /usr/bin/awk '/inet / {print $2}')"
if /usr/bin/printf "%s\n" "$lan_ips" | /usr/bin/grep -q '^192\.168\.5\.23$'; then
  configure_thunderbolt_keepalive en4 192.168.0.1 192.168.0.2
elif /usr/bin/printf "%s\n" "$lan_ips" | /usr/bin/grep -q '^192\.168\.5\.75$'; then
  configure_thunderbolt_keepalive en5 192.168.0.2 192.168.0.1
fi

if [[ -x "$PYTHON" ]]; then
  /usr/bin/pkill -f "/Users/Shared/TokenityRuntime/current/.venv/bin/python -u -m tokenity node-agent" 2>/dev/null || true
  /bin/launchctl bootout system "$NODE_AGENT_PLIST" 2>/dev/null || true
  /bin/launchctl bootstrap system "$NODE_AGENT_PLIST" 2>/dev/null || true
  /bin/launchctl enable system/ai.tokenity.node-agent 2>/dev/null || true
fi

exit 0
SCRIPT
/bin/chmod 755 "$SCRIPTS_DIR/postinstall"

echo "Building installer package..."
/usr/bin/xattr -cr "$PAYLOAD_DIR" 2>/dev/null || true
/usr/bin/dot_clean -m "$PAYLOAD_DIR" 2>/dev/null || true
pkgbuild \
  --filter '\.DS_Store$' \
  --filter '(^|/)\._[^/]*$' \
  --filter '(^|/)\.svn(/|$)' \
  --filter '(^|/)CVS(/|$)' \
  --root "$PAYLOAD_DIR" \
  --scripts "$SCRIPTS_DIR" \
  --identifier "ai.tokenity.node-agent.installer" \
  --version "$VERSION" \
  --install-location "/" \
  "$PKG_PATH"
cp "$PKG_PATH" "$DMG_ROOT/Install Tokenity Node Agent.pkg"
else
  if [[ "$NODE_AGENT_PACKAGE_MODE" == "required" ]]; then
    echo "Tokenity runtime not found. Set TOKENITY_RUNTIME_SOURCE to a local directory." >&2
    exit 1
  fi
  echo "Tokenity runtime was not found; building a controller-only drag-install DMG."
  echo "Set TOKENITY_NODE_AGENT_PACKAGE=required and TOKENITY_RUNTIME_SOURCE to include the Node Agent installer."
fi

if [[ "$BUILD_NODE_AGENT_PACKAGE" == "1" ]]; then
cat > "$DMG_ROOT/README.txt" <<README
Tokenity ${VERSION}

1. Drag TokenityControl.app onto the Applications folder.
2. Run "Install Tokenity Node Agent.pkg" on every Mac that will execute models.

The Node Agent package installs:
- /Users/Shared/TokenityCode
- /Users/Shared/TokenityRuntime
- NodeAgent LaunchDaemon on port 9100

Model weights are not included by default because they are very large.
Place compatible MLX models under:
/Users/Shared/TokenityModels

To build an offline DMG with the model included, rerun package-tokenity-dmg.sh
with TOKENITY_INCLUDE_MODEL=1.
README
else
cat > "$DMG_ROOT/README.txt" <<README
Tokenity ${VERSION}

Drag TokenityControl.app onto the Applications folder, then open it from
Applications.

This controller-only DMG does not contain the privileged Node Agent runtime.
Before running models, install a compatible Tokenity Node Agent on every Mac
that will participate in inference. The Agent listens on port 9100 and models
belong under:
/Users/Shared/TokenityModels

The first-launch guide in Tokenity explains the Cluster -> Models -> Chat
workflow.
README
fi

echo "Creating DMG..."
hdiutil create \
  -volname "Tokenity ${VERSION}" \
  -srcfolder "$DMG_ROOT" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

shasum -a 256 "$DMG_PATH" > "$DMG_CHECKSUM_PATH"

echo "Built:"
if [[ "$BUILD_NODE_AGENT_PACKAGE" == "1" ]]; then
  echo "$PKG_PATH"
fi
echo "$DMG_PATH"
echo "$DMG_CHECKSUM_PATH"
