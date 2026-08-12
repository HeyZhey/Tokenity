#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/packaging/install-layout.env"
INSTALL_ROOT="${TOKENITY_INSTALL_ROOT:-$TOKENITY_INSTALL_ROOT_DEFAULT}"
CODE_ROOT="$INSTALL_ROOT/Code"
RUNTIME_ROOT="$INSTALL_ROOT/Runtime"
MODEL_ROOT="$INSTALL_ROOT/Models"
STATE_ROOT="$INSTALL_ROOT/State"
LOG_ROOT="$INSTALL_ROOT/Logs"
RUNTIME_PYTHON="$RUNTIME_ROOT/current/.venv/bin/python"
RUNTIME_MEDIA_ROOT="${TOKENITY_RUNTIME_MEDIA_ROOT:-$TOKENITY_RUNTIME_MEDIA_ROOT_DEFAULT}"
TB_INTERFACE="${TOKENITY_TB_INTERFACE:-}"
TB_LOCAL_IP="${TOKENITY_TB_LOCAL_IP:-}"
TB_PEER_IP="${TOKENITY_TB_PEER_IP:-}"
export TOKENITY_DATA_ROOT="$INSTALL_ROOT"
export TOKENITY_CODE_ROOT="$CODE_ROOT"
export TOKENITY_MODEL_ROOT="$MODEL_ROOT"
export TOKENITY_RUNTIME_ROOT="$RUNTIME_ROOT"
export TOKENITY_STATE_ROOT="$STATE_ROOT"
export TOKENITY_LOG_ROOT="$LOG_ROOT"
export TOKENITY_RUNTIME_PYTHON="$RUNTIME_PYTHON"
if [[ "$INSTALL_ROOT" != /* || "$INSTALL_ROOT" =~ [[:space:]\<\>\&] ||
      "$INSTALL_ROOT" == *\"* || "$INSTALL_ROOT" == *\\* ]]; then
  echo "TOKENITY_INSTALL_ROOT must be an absolute path without whitespace or XML metacharacters." >&2
  exit 1
fi
if [[ "$RUNTIME_MEDIA_ROOT" != /* || "$RUNTIME_MEDIA_ROOT" == *\"* ||
      "$RUNTIME_MEDIA_ROOT" == *\\* || "$RUNTIME_MEDIA_ROOT" == *$'\n'* ||
      "$RUNTIME_MEDIA_ROOT" == *$'\r'* || "$RUNTIME_MEDIA_ROOT" == *$'\t'* ]]; then
  echo "TOKENITY_RUNTIME_MEDIA_ROOT must be an absolute path without JSON metacharacters." >&2
  exit 1
fi
VERSION="${TOKENITY_VERSION:-0.1.0}"
DIST_DIR="$ROOT/dist"
WORK_DIR="$DIST_DIR/package-work"
PAYLOAD_DIR="$WORK_DIR/payload"
SCRIPTS_DIR="$WORK_DIR/scripts"
DMG_ROOT="$WORK_DIR/dmg-root"
RUNTIME_CACHE="$DIST_DIR/runtime-cache/TokenityRuntime"
RUNTIME_LOCK="$ROOT/packaging/runtime/runtime-lock.json"
RUNTIME_MANIFEST_TOOL="$ROOT/scripts/tokenity-runtime-manifest.py"
RUNTIME_SOURCE="${TOKENITY_RUNTIME_SOURCE:-}"
NODE_AGENT_PACKAGE_MODE="${TOKENITY_NODE_AGENT_PACKAGE:-required}"
LOCK_TOKENITY_VERSION="$(
  /usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokenity_version"])' \
    "$RUNTIME_LOCK"
)"
RUNTIME_ID="$(
  /usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime_id"])' \
    "$RUNTIME_LOCK"
)"
RUNTIME_MINIMUM_MACOS="$(
  /usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["minimum_macos"])' \
    "$RUNTIME_LOCK"
)"
RUNTIME_RELEASE_TAG="$(
  /usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["release_tag"])' \
    "$RUNTIME_LOCK"
)"
RUNTIME_PACKAGE_IDENTIFIER="$(
  /usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["package_identifier"])' \
    "$RUNTIME_LOCK"
)"
RUNTIME_ARTIFACT_NAME="$(
  /usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifact_filename"])' \
    "$RUNTIME_LOCK"
)"
RUNTIME_DOWNLOAD_BASE_URL="${TOKENITY_RUNTIME_DOWNLOAD_BASE_URL:-https://github.com/HeyZhey/Tokenity/releases/download/${RUNTIME_RELEASE_TAG}}"
PKG_PATH="$DIST_DIR/$RUNTIME_ARTIFACT_NAME"
PKG_CHECKSUM_PATH="$PKG_PATH.sha256"
RUNTIME_CATALOG_PATH="$DIST_DIR/Tokenity-RuntimeCatalog-${RUNTIME_ID}.json"
DMG_PATH="$DIST_DIR/Tokenity-${VERSION}.dmg"
DMG_CHECKSUM_PATH="$DMG_PATH.sha256"
APP_BUNDLE="$DMG_ROOT/TokenityControl.app"
BACKEND_SOURCE="$ROOT/tokenity/serving/distributed_openai.py"

if [[ "$VERSION" != "$LOCK_TOKENITY_VERSION" ]]; then
  echo "TOKENITY_VERSION=$VERSION does not match the Runtime lock version $LOCK_TOKENITY_VERSION." >&2
  exit 1
fi

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
  --exclude "__pycache__"
  --exclude "*.pyc"
  --exclude "*.pyo"
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

runtime_is_available() {
  local root="$1"
  [[ -d "$root" && -x "$root/current/.venv/bin/python" ]]
}

if [[ -z "$RUNTIME_SOURCE" ]]; then
  if runtime_is_available "$RUNTIME_ROOT"; then
    RUNTIME_SOURCE="$RUNTIME_ROOT"
  else
    RUNTIME_SOURCE="$RUNTIME_CACHE"
  fi
fi

if [[ -n "$TB_INTERFACE$TB_LOCAL_IP$TB_PEER_IP" ]] &&
   [[ -z "$TB_INTERFACE" || -z "$TB_LOCAL_IP" || -z "$TB_PEER_IP" ]]; then
  echo "TOKENITY_TB_INTERFACE, TOKENITY_TB_LOCAL_IP, and TOKENITY_TB_PEER_IP must be set together." >&2
  exit 1
fi

case "$NODE_AGENT_PACKAGE_MODE" in
  auto)
    if runtime_is_available "$RUNTIME_SOURCE" || runtime_is_available "$RUNTIME_CACHE"; then
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
rm -f \
  "$PKG_PATH" \
  "$PKG_CHECKSUM_PATH" \
  "$RUNTIME_CATALOG_PATH" \
  "$DMG_PATH" \
  "$DMG_CHECKSUM_PATH"
mkdir -p "$SCRIPTS_DIR" "$DMG_ROOT" "$DIST_DIR"

echo "Building TokenityControl.app..."
TOKENITY_BUILD_CONFIGURATION=release \
TOKENITY_APP_BUNDLE_PATH="$APP_BUNDLE" \
TOKENITY_BUNDLE_IDENTIFIER=ai.tokenity.control \
"$ROOT/scripts/build-tokenity-control-app.sh" >/dev/null

cat > "$APP_BUNDLE/Contents/Resources/DeploymentConfiguration.json" <<JSON
{
  "TOKENITY_DATA_ROOT": "$INSTALL_ROOT",
  "TOKENITY_CODE_ROOT": "$CODE_ROOT",
  "TOKENITY_MODEL_ROOT": "$MODEL_ROOT",
  "TOKENITY_RUNTIME_ROOT": "$RUNTIME_ROOT",
  "TOKENITY_RUNTIME_PYTHON": "$RUNTIME_PYTHON",
  "TOKENITY_RUNTIME_MEDIA_ROOT": "$RUNTIME_MEDIA_ROOT",
  "TOKENITY_STATE_ROOT": "$STATE_ROOT",
  "TOKENITY_LOG_ROOT": "$LOG_ROOT"
}
JSON
codesign --force --deep --sign - "$APP_BUNDLE" >/dev/null
codesign --verify --deep --strict "$APP_BUNDLE"
ln -s /Applications "$DMG_ROOT/Applications"

if [[ "$BUILD_NODE_AGENT_PACKAGE" == "1" ]]; then
mkdir -p "$PAYLOAD_DIR$CODE_ROOT" \
  "$PAYLOAD_DIR$RUNTIME_ROOT" \
  "$PAYLOAD_DIR$MODEL_ROOT" \
  "$PAYLOAD_DIR/Library/LaunchDaemons" \
  "$PAYLOAD_DIR/usr/local/bin"
/usr/bin/install -m 755 "$ROOT/scripts/uninstall-tokenity-node-agent.sh" \
  "$PAYLOAD_DIR/usr/local/bin/tokenity-uninstall-node-agent"

echo "Copying Tokenity backend code..."
copy_dir "$ROOT/" "$PAYLOAD_DIR$CODE_ROOT"

echo "Preparing Tokenity runtime..."
if runtime_is_available "$RUNTIME_SOURCE" && [[ "$RUNTIME_SOURCE" != "$RUNTIME_CACHE" ]]; then
  copy_runtime_dir "$RUNTIME_SOURCE/" "$RUNTIME_CACHE"
elif runtime_is_available "$RUNTIME_CACHE"; then
  echo "Using the cached Tokenity runtime at $RUNTIME_CACHE."
else
  echo "Tokenity runtime not found or its Python is not executable." >&2
  echo "Run scripts/import-tokenity-runtime.sh first, or set TOKENITY_RUNTIME_SOURCE." >&2
  exit 1
fi
"$RUNTIME_MANIFEST_TOOL" prepare "$RUNTIME_CACHE" \
  --lock "$RUNTIME_LOCK" \
  --output "$RUNTIME_CACHE/runtime-manifest.json"
"$RUNTIME_MANIFEST_TOOL" verify "$RUNTIME_CACHE" \
  --lock "$RUNTIME_LOCK" \
  --manifest "$RUNTIME_CACHE/runtime-manifest.json"
copy_runtime_dir "$RUNTIME_CACHE/" "$PAYLOAD_DIR$RUNTIME_ROOT"

if [[ "${TOKENITY_INCLUDE_MODEL:-0}" == "1" ]]; then
  MODEL_NAME="${TOKENITY_MODEL_NAME:-Qwen3.5-122B-A10B-4bit}"
  MODEL_SOURCE="${TOKENITY_MODEL_SOURCE:-$MODEL_ROOT/${MODEL_NAME}}"
  MODEL_DEST="$PAYLOAD_DIR$MODEL_ROOT/$MODEL_NAME"
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

cat > "$PAYLOAD_DIR/Library/LaunchDaemons/ai.tokenity.node-agent.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ai.tokenity.node-agent</string>
  <key>ProgramArguments</key>
  <array>
    <string>$RUNTIME_PYTHON</string>
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
    <string>$RUNTIME_ROOT/current/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>PYTHONPATH</key>
    <string>$CODE_ROOT</string>
    <key>TOKENITY_INSTANCE_STATE_ROOT</key>
    <string>$STATE_ROOT/instances</string>
    <key>TOKENITY_DATA_ROOT</key>
    <string>$INSTALL_ROOT</string>
    <key>TOKENITY_MODEL_ROOT</key>
    <string>$MODEL_ROOT</string>
    <key>TOKENITY_RUNTIME_ROOT</key>
    <string>$RUNTIME_ROOT</string>
    <key>TOKENITY_RUNTIME_PYTHON</key>
    <string>$RUNTIME_PYTHON</string>
    <key>TOKENITY_CODE_ROOT</key>
    <string>$CODE_ROOT</string>
    <key>TOKENITY_STATE_ROOT</key>
    <string>$STATE_ROOT</string>
    <key>TOKENITY_LOG_ROOT</key>
    <string>$LOG_ROOT</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ProcessType</key>
  <string>Interactive</string>
  <key>StandardOutPath</key>
  <string>$LOG_ROOT/node-agent.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_ROOT/node-agent.err</string>
</dict>
</plist>
PLIST

cat > "$PAYLOAD_DIR/Library/LaunchDaemons/ai.tokenity.node-agent-watchdog.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ai.tokenity.node-agent-watchdog</string>
  <key>ProgramArguments</key>
  <array>
    <string>$RUNTIME_PYTHON</string>
    <string>-u</string>
    <string>-m</string>
    <string>tokenity.node_agent.watchdog</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>$RUNTIME_ROOT/current/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>PYTHONPATH</key>
    <string>$CODE_ROOT</string>
    <key>TOKENITY_DATA_ROOT</key>
    <string>$INSTALL_ROOT</string>
    <key>TOKENITY_STATE_ROOT</key>
    <string>$STATE_ROOT</string>
    <key>TOKENITY_LOG_ROOT</key>
    <string>$LOG_ROOT</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ProcessType</key>
  <string>Background</string>
  <key>ThrottleInterval</key>
  <integer>10</integer>
  <key>StandardOutPath</key>
  <string>$LOG_ROOT/node-agent-watchdog.stdout.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_ROOT/node-agent-watchdog.stderr.log</string>
</dict>
</plist>
PLIST

RUNTIME_SIZE_KB="$(du -sk "$RUNTIME_CACHE" | awk '{print $1}')"
{
  printf '#!/bin/zsh\nset -eu\n'
  printf 'MINIMUM_MACOS="%s"\n' "$RUNTIME_MINIMUM_MACOS"
  printf 'REQUIRED_KB="%s"\n' "$((RUNTIME_SIZE_KB + 1048576))"
  printf 'INSTALL_ROOT=%q\n' "$INSTALL_ROOT"
  printf 'RUNTIME_PYTHON=%q\n' "$RUNTIME_PYTHON"
  printf 'STATE_ROOT=%q\n' "$STATE_ROOT"
  cat <<'SCRIPT'

NODE_AGENT_PLIST="/Library/LaunchDaemons/ai.tokenity.node-agent.plist"
WATCHDOG_PLIST="/Library/LaunchDaemons/ai.tokenity.node-agent-watchdog.plist"
MAINTENANCE_PATH="$STATE_ROOT/maintenance.json"
AGENT_URL="http://127.0.0.1:9100"

remove_legacy_user_agents() {
  local home_directory legacy_plist legacy_uid
  while IFS= read -r home_directory; do
    legacy_plist="$home_directory/Library/LaunchAgents/local.tokenity.node-agent.plist"
    [[ -f "$legacy_plist" ]] || continue
    legacy_uid="$(/usr/bin/stat -f '%u' "$legacy_plist" 2>/dev/null || true)"
    if [[ -n "$legacy_uid" ]]; then
      /bin/launchctl bootout \
        "gui/$legacy_uid/local.tokenity.node-agent" 2>/dev/null || \
        /bin/launchctl bootout "gui/$legacy_uid" "$legacy_plist" 2>/dev/null || true
    fi
    /bin/rm -f "$legacy_plist"
  done < <(/usr/bin/dscacheutil -q user | /usr/bin/awk '/^dir: / {sub(/^dir: /, ""); print}')
}

version_at_least() {
  /usr/bin/awk -v observed="$1" -v required="$2" 'BEGIN {
    observed_count = split(observed, observed_parts, ".")
    required_count = split(required, required_parts, ".")
    count = observed_count > required_count ? observed_count : required_count
    for (index = 1; index <= count; index++) {
      observed_value = observed_parts[index] + 0
      required_value = required_parts[index] + 0
      if (observed_value > required_value) exit 0
      if (observed_value < required_value) exit 1
    }
    exit 0
  }'
}

if [[ "$(/usr/bin/uname -m)" != "arm64" ]]; then
  echo "Tokenity Runtime requires an Apple-silicon Mac (arm64)." >&2
  exit 1
fi

observed_macos="$(/usr/bin/sw_vers -productVersion)"
if ! version_at_least "$observed_macos" "$MINIMUM_MACOS"; then
  echo "This Tokenity Runtime requires macOS $MINIMUM_MACOS or newer; found $observed_macos." >&2
  exit 1
fi

available_kb="$(/bin/df -Pk "$(/usr/bin/dirname "$INSTALL_ROOT")" 2>/dev/null | /usr/bin/awk 'NR == 2 {print $4}')"
if [[ -n "$available_kb" && "$available_kb" -lt "$REQUIRED_KB" ]]; then
  echo "Tokenity Runtime needs at least $REQUIRED_KB KB free under $INSTALL_ROOT." >&2
  exit 1
fi

/bin/mkdir -p "$STATE_ROOT"
/usr/bin/printf '{"until":%s,"reason":"package_update"}\n' \
  "$(( $(/bin/date +%s) + 600 ))" > "$MAINTENANCE_PATH"
/bin/launchctl bootout system "$WATCHDOG_PLIST" 2>/dev/null || true

if /usr/bin/curl --noproxy '*' --silent --fail --max-time 3 \
  "$AGENT_URL/v1/node/info" >/dev/null; then
  /usr/bin/curl --noproxy '*' --silent --show-error --max-time 5 \
    -H 'Content-Type: application/json' \
    -d '{}' \
    "$AGENT_URL/v1/node/request-stop-all" >/dev/null 2>&1 || true
  /usr/bin/curl --noproxy '*' --silent --show-error --max-time 40 \
    -H 'Content-Type: application/json' \
    -d '{"timeout":30}' \
    "$AGENT_URL/v1/node/stop-all" >/dev/null 2>&1 || true
fi

remove_legacy_user_agents

for _ in {1..45}; do
  if ! /usr/bin/pgrep -f 'tokenity distributed-openai serve' >/dev/null 2>&1; then
    break
  fi
  /bin/sleep 1
done
if /usr/bin/pgrep -f 'tokenity distributed-openai serve' >/dev/null 2>&1; then
  echo "A Tokenity model runtime is still active; refusing to overwrite it." >&2
  exit 1
fi

/bin/launchctl bootout system "$NODE_AGENT_PLIST" 2>/dev/null || true
/usr/bin/pkill -f \
  "$RUNTIME_PYTHON -u -m tokenity node-agent" \
  2>/dev/null || true

exit 0
SCRIPT
} > "$SCRIPTS_DIR/preinstall"

RUNTIME_EXPECTED_MLX="$(
  /usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["packages"]["mlx"])' \
    "$RUNTIME_LOCK"
)"
RUNTIME_EXPECTED_MLX_LM="$(
  /usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["packages"]["mlx-lm"])' \
    "$RUNTIME_LOCK"
)"
{
  printf '#!/bin/zsh\nset -eu\n'
  printf 'EXPECTED_MLX="%s"\n' "$RUNTIME_EXPECTED_MLX"
  printf 'EXPECTED_MLX_LM="%s"\n' "$RUNTIME_EXPECTED_MLX_LM"
  printf 'INSTALL_ROOT=%q\n' "$INSTALL_ROOT"
  printf 'CODE_ROOT=%q\n' "$CODE_ROOT"
  printf 'RUNTIME_ROOT=%q\n' "$RUNTIME_ROOT"
  printf 'MODEL_ROOT=%q\n' "$MODEL_ROOT"
  printf 'STATE_ROOT=%q\n' "$STATE_ROOT"
  printf 'LOG_ROOT=%q\n' "$LOG_ROOT"
  printf 'RUNTIME_PYTHON=%q\n' "$RUNTIME_PYTHON"
  printf 'TB_INTERFACE=%q\n' "$TB_INTERFACE"
  printf 'TB_LOCAL_IP=%q\n' "$TB_LOCAL_IP"
  printf 'TB_PEER_IP=%q\n' "$TB_PEER_IP"
  cat <<'SCRIPT'

NODE_AGENT_PLIST="/Library/LaunchDaemons/ai.tokenity.node-agent.plist"
WATCHDOG_PLIST="/Library/LaunchDaemons/ai.tokenity.node-agent-watchdog.plist"
MAINTENANCE_PATH="$STATE_ROOT/maintenance.json"
TB_PLIST="/Library/LaunchDaemons/ai.tokenity.thunderbolt-keepalive.plist"
TB_SCRIPT="/usr/local/bin/tokenity-tb-keepalive"
PYTHON="$RUNTIME_PYTHON"
AGENT_USER="$(/usr/bin/stat -f '%Su' /dev/console 2>/dev/null || true)"

if [[ -n "$AGENT_USER" && "$AGENT_USER" != "root" && "$AGENT_USER" != "loginwindow" ]]; then
  /usr/libexec/PlistBuddy -c "Add :UserName string $AGENT_USER" "$NODE_AGENT_PLIST" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c "Set :UserName $AGENT_USER" "$NODE_AGENT_PLIST"
fi

/bin/mkdir -p "$LOG_ROOT" "$MODEL_ROOT" "$STATE_ROOT/instances" /usr/local/bin
/usr/sbin/chown root:wheel "$NODE_AGENT_PLIST" 2>/dev/null || true
/usr/sbin/chown root:wheel "$WATCHDOG_PLIST" 2>/dev/null || true
if [[ -n "$AGENT_USER" && "$AGENT_USER" != "root" && "$AGENT_USER" != "loginwindow" ]]; then
  /usr/sbin/chown -R "$AGENT_USER":staff \
    "$LOG_ROOT" "$MODEL_ROOT" "$STATE_ROOT"
else
  /usr/sbin/chown -R root:wheel "$LOG_ROOT" "$STATE_ROOT"
fi
/bin/chmod 644 "$NODE_AGENT_PLIST"
/bin/chmod 644 "$WATCHDOG_PLIST"
/bin/chmod -R a+rX "$CODE_ROOT" "$RUNTIME_ROOT" "$MODEL_ROOT" 2>/dev/null || true

cat > "$TB_SCRIPT" <<'KEEPALIVE'
#!/bin/zsh
iface="$1"
local_ip="$2"
peer_ip="$3"
log_root="$4"
failures=0
reset_link() {
  /bin/date "+%Y-%m-%d %H:%M:%S resetting $iface ($1)" >> "$log_root/thunderbolt-keepalive.log"
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
    <string>$LOG_ROOT</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$LOG_ROOT/thunderbolt-keepalive.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_ROOT/thunderbolt-keepalive.err</string>
</dict>
</plist>
PLIST
  /usr/sbin/chown root:wheel "$TB_PLIST"
  /bin/chmod 644 "$TB_PLIST"
  /bin/launchctl bootout system "$TB_PLIST" 2>/dev/null || true
  /bin/launchctl bootstrap system "$TB_PLIST" 2>/dev/null || true
  /bin/launchctl enable system/ai.tokenity.thunderbolt-keepalive 2>/dev/null || true
}

if [[ -n "$TB_INTERFACE" ]]; then
  configure_thunderbolt_keepalive "$TB_INTERFACE" "$TB_LOCAL_IP" "$TB_PEER_IP"
fi

if [[ -x "$PYTHON" ]]; then
  TOKENITY_EXPECTED_MLX="$EXPECTED_MLX" \
  TOKENITY_EXPECTED_MLX_LM="$EXPECTED_MLX_LM" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$CODE_ROOT" \
  "$PYTHON" - <<'PY'
from importlib.metadata import version

import mlx.core as mx
import mlx_lm
import tokenity

import os

required = {
    "mlx": os.environ["TOKENITY_EXPECTED_MLX"],
    "mlx-lm": os.environ["TOKENITY_EXPECTED_MLX_LM"],
}
observed = {name: version(name) for name in required}
if observed != required:
    raise SystemExit(f"Runtime package mismatch: expected {required}, found {observed}")
mx.eval(mx.array([1], dtype=mx.int32))
print(f"Validated installed Tokenity Runtime: {observed}")
PY
  /usr/bin/pkill -f "$RUNTIME_PYTHON -u -m tokenity node-agent" 2>/dev/null || true
  /bin/launchctl bootout system "$NODE_AGENT_PLIST" 2>/dev/null || true
  /bin/launchctl bootstrap system "$NODE_AGENT_PLIST" 2>/dev/null || true
  /bin/launchctl enable system/ai.tokenity.node-agent 2>/dev/null || true
  healthy=0
  for _ in {1..30}; do
    if /usr/bin/curl --noproxy '*' --silent --fail --max-time 2 \
      http://127.0.0.1:9100/v1/node/health | /usr/bin/grep -q '"status":"healthy"\|"status": "healthy"'; then
      healthy=1
      break
    fi
    /bin/sleep 1
  done
  if [[ "$healthy" != "1" ]]; then
    echo "The installed Node Agent did not become healthy." >&2
    exit 1
  fi
  /bin/rm -f "$MAINTENANCE_PATH"
  /bin/launchctl bootout system "$WATCHDOG_PLIST" 2>/dev/null || true
  /bin/launchctl bootstrap system "$WATCHDOG_PLIST" 2>/dev/null || true
  /bin/launchctl enable system/ai.tokenity.node-agent-watchdog 2>/dev/null || true
  for _ in {1..10}; do
    if /usr/bin/curl --noproxy '*' --silent --fail --max-time 2 \
      http://127.0.0.1:9101/v1/watchdog/status >/dev/null; then
      exit 0
    fi
    /bin/sleep 1
  done
  echo "The Node Agent watchdog did not publish its status endpoint." >&2
  exit 1
fi

exit 0
SCRIPT
} > "$SCRIPTS_DIR/postinstall"
/bin/chmod 755 "$SCRIPTS_DIR/preinstall" "$SCRIPTS_DIR/postinstall"

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
  --identifier "$RUNTIME_PACKAGE_IDENTIFIER" \
  --version "$VERSION" \
  --install-location "/" \
  "$PKG_PATH"

(
  cd "$DIST_DIR"
  shasum -a 256 "$RUNTIME_ARTIFACT_NAME" > "$(basename "$PKG_CHECKSUM_PATH")"
)
RUNTIME_DOWNLOAD_URL="${RUNTIME_DOWNLOAD_BASE_URL%/}/$RUNTIME_ARTIFACT_NAME"
"$RUNTIME_MANIFEST_TOOL" catalog \
  --lock "$RUNTIME_LOCK" \
  --runtime-manifest "$RUNTIME_CACHE/runtime-manifest.json" \
  --artifact "$PKG_PATH" \
  --download-url "$RUNTIME_DOWNLOAD_URL" \
  --output "$RUNTIME_CATALOG_PATH"
"$RUNTIME_MANIFEST_TOOL" verify-artifact \
  --catalog "$RUNTIME_CATALOG_PATH" \
  --artifact "$PKG_PATH"

cp "$PKG_PATH" "$APP_BUNDLE/Contents/Resources/$RUNTIME_ARTIFACT_NAME"
cp "$RUNTIME_CATALOG_PATH" "$APP_BUNDLE/Contents/Resources/RuntimeCatalog.json"
codesign --force --deep --sign - "$APP_BUNDLE" >/dev/null
codesign --verify --deep --strict "$APP_BUNDLE"

ln -s \
  "TokenityControl.app/Contents/Resources/$RUNTIME_ARTIFACT_NAME" \
  "$DMG_ROOT/Install Tokenity Node Agent.pkg"
cp "$PKG_CHECKSUM_PATH" "$DMG_ROOT/Runtime Installer.sha256"
cp "$RUNTIME_CATALOG_PATH" "$DMG_ROOT/Runtime Catalog.json"
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
- ${CODE_ROOT}
- ${RUNTIME_ROOT}
- NodeAgent LaunchDaemon on port 9100

This Runtime is validated for Apple silicon and requires macOS
${RUNTIME_MINIMUM_MACOS} or newer. The same verified installer remains embedded
inside TokenityControl.app after the app is copied to Applications.

Model weights are not included by default because they are very large.
Place compatible MLX models under:
${MODEL_ROOT}

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
${MODEL_ROOT}

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

(
  cd "$DIST_DIR"
  shasum -a 256 "$(basename "$DMG_PATH")" > "$(basename "$DMG_CHECKSUM_PATH")"
)

echo "Built:"
if [[ "$BUILD_NODE_AGENT_PACKAGE" == "1" ]]; then
  echo "$PKG_PATH"
  echo "$PKG_CHECKSUM_PATH"
  echo "$RUNTIME_CATALOG_PATH"
fi
echo "$DMG_PATH"
echo "$DMG_CHECKSUM_PATH"
