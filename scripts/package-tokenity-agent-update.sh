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
INSTALLED_RUNTIME="$RUNTIME_ROOT/current/.venv/bin/python"
if [[ "$INSTALL_ROOT" != /* || "$INSTALL_ROOT" =~ [[:space:]\<\>\&] ||
      "$INSTALL_ROOT" == *\"* || "$INSTALL_ROOT" == *\\* ]]; then
  echo "TOKENITY_INSTALL_ROOT must be an absolute path without whitespace or XML metacharacters." >&2
  exit 1
fi
VERSION="${TOKENITY_VERSION:-0.1.0}"
DIST_DIR="$ROOT/dist"
WORK_DIR="$DIST_DIR/agent-update-work"
PAYLOAD_DIR="$WORK_DIR/payload"
SCRIPTS_DIR="$WORK_DIR/scripts"
PKG_PATH="$DIST_DIR/Tokenity-NodeAgent-${VERSION}.pkg"
LAUNCHD_LABEL="ai.tokenity.node-agent"
RUNTIME_LOCK="$ROOT/packaging/runtime/runtime-lock.json"
EXPECTED_MLX="$(
  /usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["packages"]["mlx"])' \
    "$RUNTIME_LOCK"
)"
EXPECTED_MLX_LM="$(
  /usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["packages"]["mlx-lm"])' \
    "$RUNTIME_LOCK"
)"

if [[ ! -f "$ROOT/tokenity/node_agent/agent.py" ]] ||
   [[ ! -f "$ROOT/tokenity/serving/distributed_openai.py" ]]; then
  echo "Tokenity Node Agent sources are incomplete." >&2
  exit 1
fi

EXPECTED_REVISION="$(
  {
    /usr/bin/printf '%s' "agent.py"
    /bin/cat "$ROOT/tokenity/node_agent/agent.py"
    /usr/bin/printf '%s' "distributed_openai.py"
    /bin/cat "$ROOT/tokenity/serving/distributed_openai.py"
  } | /usr/bin/shasum -a 256 | /usr/bin/awk '{print $1}'
)"

/bin/rm -rf "$WORK_DIR"
/bin/mkdir -p "$PAYLOAD_DIR$CODE_ROOT" \
  "$PAYLOAD_DIR/Library/LaunchDaemons" "$PAYLOAD_DIR/usr/local/bin" \
  "$SCRIPTS_DIR" "$DIST_DIR"
/usr/bin/install -m 755 "$ROOT/scripts/uninstall-tokenity-node-agent.sh" \
  "$PAYLOAD_DIR/usr/local/bin/tokenity-uninstall-node-agent"

/usr/bin/rsync -a \
  --delete \
  --exclude '.DS_Store' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$ROOT/tokenity/" \
  "$PAYLOAD_DIR$CODE_ROOT/tokenity/"

/bin/cat > "$PAYLOAD_DIR/Library/LaunchDaemons/ai.tokenity.node-agent-watchdog.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ai.tokenity.node-agent-watchdog</string>
  <key>ProgramArguments</key>
  <array>
    <string>$INSTALLED_RUNTIME</string>
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

{
  /usr/bin/printf '#!/bin/zsh\nset -eu\n'
  /usr/bin/printf 'EXPECTED_MLX="%s"\n' "$EXPECTED_MLX"
  /usr/bin/printf 'EXPECTED_MLX_LM="%s"\n' "$EXPECTED_MLX_LM"
  /usr/bin/printf 'PYTHON="%s"\n' "$INSTALLED_RUNTIME"
  /usr/bin/printf 'STATE_ROOT="%s"\n' "$STATE_ROOT"
  /bin/cat <<'SCRIPT'

PLIST="/Library/LaunchDaemons/ai.tokenity.node-agent.plist"
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

if [[ ! -x "$PYTHON" ]]; then
  echo "Tokenity Runtime is missing at $PYTHON; refusing a code-only Agent update." >&2
  exit 1
fi
if [[ ! -f "$PLIST" ]]; then
  echo "The installer-managed Node Agent LaunchDaemon is missing at $PLIST." >&2
  exit 1
fi

/bin/mkdir -p "$STATE_ROOT"
/usr/bin/printf '{"until":%s,"reason":"agent_update"}\n' \
  "$(( $(/bin/date +%s) + 600 ))" > "$MAINTENANCE_PATH"
/bin/launchctl bootout system "$WATCHDOG_PLIST" 2>/dev/null || true

TOKENITY_EXPECTED_MLX="$EXPECTED_MLX" \
TOKENITY_EXPECTED_MLX_LM="$EXPECTED_MLX_LM" \
"$PYTHON" - <<'PY'
from importlib.metadata import version
import os

required = {
    "mlx": os.environ["TOKENITY_EXPECTED_MLX"],
    "mlx-lm": os.environ["TOKENITY_EXPECTED_MLX_LM"],
}
observed = {name: version(name) for name in required}
if observed != required:
    raise SystemExit(f"Runtime package mismatch: expected {required}, found {observed}")
print(f"Validated installed Tokenity Runtime: {observed}")
PY

if /usr/bin/curl --noproxy '*' --silent --fail --max-time 3 \
  "$AGENT_URL/v1/node/info" >/dev/null; then
  /usr/bin/curl --noproxy '*' --silent --show-error --max-time 5 \
    -H 'Content-Type: application/json' \
    -d '{}' \
    "$AGENT_URL/v1/node/request-stop-all" >/dev/null
  /usr/bin/curl --noproxy '*' --silent --show-error --max-time 40 \
    -H 'Content-Type: application/json' \
    -d '{"timeout":30}' \
    "$AGENT_URL/v1/node/stop-all" >/dev/null
fi

remove_legacy_user_agents

for _ in {1..45}; do
  if ! /usr/bin/pgrep -f 'tokenity distributed-openai serve' >/dev/null 2>&1; then
    exit 0
  fi
  /bin/sleep 1
done

echo "A Tokenity model runtime is still active; refusing to orphan it during Agent upgrade." >&2
exit 1
SCRIPT
} > "$SCRIPTS_DIR/preinstall"

/bin/cat > "$SCRIPTS_DIR/postinstall" <<SCRIPT
#!/bin/zsh
set -eu

EXPECTED_REVISION="$EXPECTED_REVISION"
AGENT_URL="http://127.0.0.1:9100"
LABEL="$LAUNCHD_LABEL"
PLIST="/Library/LaunchDaemons/ai.tokenity.node-agent.plist"
WATCHDOG_PLIST="/Library/LaunchDaemons/ai.tokenity.node-agent-watchdog.plist"
INSTALL_ROOT="$INSTALL_ROOT"
CODE_ROOT="$CODE_ROOT"
RUNTIME_ROOT="$RUNTIME_ROOT"
MODEL_ROOT="$MODEL_ROOT"
STATE_ROOT="$STATE_ROOT"
LOG_ROOT="$LOG_ROOT"
RUNTIME_PYTHON="$INSTALLED_RUNTIME"
MAINTENANCE_PATH="$STATE_ROOT/maintenance.json"
AGENT_USER="\$(/usr/bin/stat -f '%Su' /dev/console 2>/dev/null || true)"

set_plist_environment() {
  local key="\$1"
  local value="\$2"
  /usr/libexec/PlistBuddy -c "Add :EnvironmentVariables:\$key string \$value" "\$PLIST" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c "Set :EnvironmentVariables:\$key \$value" "\$PLIST"
}

/bin/chmod -R a+rX "\$CODE_ROOT"
/usr/bin/find "\$CODE_ROOT/tokenity" -type d -name __pycache__ -prune -exec /bin/rm -rf {} + 2>/dev/null || true
/bin/mkdir -p "\$LOG_ROOT" "\$STATE_ROOT/instances"
set_plist_environment PYTHONPATH "\$CODE_ROOT"
set_plist_environment TOKENITY_DATA_ROOT "\$INSTALL_ROOT"
set_plist_environment TOKENITY_CODE_ROOT "\$CODE_ROOT"
set_plist_environment TOKENITY_MODEL_ROOT "\$MODEL_ROOT"
set_plist_environment TOKENITY_RUNTIME_ROOT "\$RUNTIME_ROOT"
set_plist_environment TOKENITY_RUNTIME_PYTHON "\$RUNTIME_PYTHON"
set_plist_environment TOKENITY_STATE_ROOT "\$STATE_ROOT"
set_plist_environment TOKENITY_LOG_ROOT "\$LOG_ROOT"
set_plist_environment TOKENITY_INSTANCE_STATE_ROOT "\$STATE_ROOT/instances"
/usr/sbin/chown root:wheel "\$PLIST" "\$WATCHDOG_PLIST"
/bin/chmod 644 "\$PLIST" "\$WATCHDOG_PLIST"
if [[ -n "\$AGENT_USER" && "\$AGENT_USER" != "root" && "\$AGENT_USER" != "loginwindow" ]]; then
  /usr/sbin/chown -R "\$AGENT_USER":staff "\$LOG_ROOT" "\$STATE_ROOT"
fi

/bin/launchctl kickstart -k "system/\$LABEL"

healthy=0
for _ in {1..30}; do
  payload="\$(/usr/bin/curl --noproxy '*' --silent --max-time 2 "\$AGENT_URL/v1/node/health" 2>/dev/null || true)"
  if [[ "\$payload" == *"\$EXPECTED_REVISION"* ]] &&
     [[ "\$payload" == *"healthy"* ]]; then
    healthy=1
    break
  fi
  /bin/sleep 1
done

if [[ "\$healthy" != "1" ]]; then
  echo "The updated Node Agent did not become healthy." >&2
  exit 1
fi

payload="\$(/usr/bin/curl --noproxy '*' --silent --max-time 2 "\$AGENT_URL/v1/node/info" 2>/dev/null || true)"
if [[ "\$payload" != *"managed_instances"* ]] || [[ "\$payload" != *"\$EXPECTED_REVISION"* ]]; then
  echo "The updated Node Agent did not publish the expected revision and managed-instance contract." >&2
  exit 1
fi

/bin/rm -f "\$MAINTENANCE_PATH"
/bin/launchctl bootout system "\$WATCHDOG_PLIST" 2>/dev/null || true
/bin/launchctl bootstrap system "\$WATCHDOG_PLIST"
/bin/launchctl enable system/ai.tokenity.node-agent-watchdog 2>/dev/null || true
for _ in {1..10}; do
  if /usr/bin/curl --noproxy '*' --silent --fail --max-time 2 \
    http://127.0.0.1:9101/v1/watchdog/status >/dev/null; then
    echo "Tokenity Node Agent updated to \$EXPECTED_REVISION with watchdog health reporting."
    exit 0
  fi
  /bin/sleep 1
done

echo "The updated Node Agent watchdog did not publish status." >&2
exit 1
SCRIPT

/bin/chmod 755 "$SCRIPTS_DIR/preinstall" "$SCRIPTS_DIR/postinstall"
/usr/bin/xattr -cr "$PAYLOAD_DIR" 2>/dev/null || true
/usr/bin/dot_clean -m "$PAYLOAD_DIR" 2>/dev/null || true

/usr/bin/pkgbuild \
  --filter '\.DS_Store$' \
  --filter '(^|/)\._[^/]*$' \
  --root "$PAYLOAD_DIR" \
  --scripts "$SCRIPTS_DIR" \
  --identifier "ai.tokenity.node-agent.update" \
  --version "$VERSION" \
  --install-location "/" \
  "$PKG_PATH"

echo "Built $PKG_PATH"
echo "Expected Agent code revision: $EXPECTED_REVISION"
