#!/bin/zsh
set -eu

if [[ "$#" -lt 3 || "$#" -gt 5 ]]; then
  echo "Usage: $0 WORKTREE LAUNCHD_DOMAIN STATE_ROOT [start|stop] [PYTHON]" >&2
  exit 2
fi

WORKTREE="$1"
LAUNCHD_DOMAIN="$2"
STATE_ROOT="$3"
ACTION="${4:-start}"
PYTHON="${5:-${TOKENITY_RUNTIME_PYTHON:-$(command -v python3)}}"
PYTHON_BIN_DIRECTORY="$(dirname "$PYTHON")"
AGENT_LABEL="ai.tokenity.node-agent-isolated"
WATCHDOG_LABEL="ai.tokenity.node-agent-watchdog-isolated"
AGENT_PLIST="$STATE_ROOT/$AGENT_LABEL.plist"
WATCHDOG_PLIST="$STATE_ROOT/$WATCHDOG_LABEL.plist"

stop_jobs() {
  /bin/launchctl bootout "$LAUNCHD_DOMAIN/$WATCHDOG_LABEL" 2>/dev/null || true
  /bin/launchctl bootout "$LAUNCHD_DOMAIN/$AGENT_LABEL" 2>/dev/null || true
}

if [[ "$ACTION" == "stop" ]]; then
  stop_jobs
  exit 0
fi
if [[ "$ACTION" != "start" ]]; then
  echo "Action must be start or stop." >&2
  exit 2
fi
if [[ ! -x "$PYTHON" || ! -f "$WORKTREE/tokenity/node_agent/agent.py" ]]; then
  echo "Isolated Tokenity code or shared runtime is missing." >&2
  exit 1
fi

/bin/mkdir -p "$STATE_ROOT/instances" "$STATE_ROOT/logs"

/bin/cat > "$AGENT_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$AGENT_LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON</string><string>-u</string><string>-m</string>
    <string>tokenity</string><string>node-agent</string>
    <string>--host</string><string>0.0.0.0</string>
    <string>--port</string><string>9200</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>$PYTHON_BIN_DIRECTORY:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>PYTHONPATH</key><string>$WORKTREE</string>
    <key>TOKENITY_INSTANCE_STATE_ROOT</key><string>$STATE_ROOT/instances</string>
    <key>TOKENITY_DISABLE_UNJOURNALED_ORPHAN_SCAN</key><string>1</string>
  </dict>
  <key>WorkingDirectory</key><string>$WORKTREE</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>5</integer>
  <key>StandardOutPath</key><string>$STATE_ROOT/logs/agent.stdout.log</string>
  <key>StandardErrorPath</key><string>$STATE_ROOT/logs/agent.stderr.log</string>
</dict>
</plist>
PLIST

/bin/cat > "$WATCHDOG_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$WATCHDOG_LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON</string><string>-u</string><string>-m</string>
    <string>tokenity.node_agent.watchdog</string>
    <string>--agent-port</string><string>9200</string>
    <string>--status-port</string><string>9201</string>
    <string>--launchd-label</string><string>$AGENT_LABEL</string>
    <string>--launchd-domain</string><string>$LAUNCHD_DOMAIN</string>
    <string>--state-path</string><string>$STATE_ROOT/watchdog.json</string>
    <string>--maintenance-path</string><string>$STATE_ROOT/maintenance.json</string>
    <string>--log-path</string><string>$STATE_ROOT/logs/watchdog.log</string>
    <string>--lock-path</string><string>$STATE_ROOT/watchdog.lock</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>$PYTHON_BIN_DIRECTORY:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>PYTHONPATH</key><string>$WORKTREE</string>
  </dict>
  <key>WorkingDirectory</key><string>$WORKTREE</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>5</integer>
  <key>StandardOutPath</key><string>$STATE_ROOT/logs/watchdog.stdout.log</string>
  <key>StandardErrorPath</key><string>$STATE_ROOT/logs/watchdog.stderr.log</string>
</dict>
</plist>
PLIST

/bin/chmod 600 "$AGENT_PLIST" "$WATCHDOG_PLIST"
stop_jobs
for _ in {1..20}; do
  if ! /bin/launchctl print "$LAUNCHD_DOMAIN/$AGENT_LABEL" >/dev/null 2>&1 && \
     ! /bin/launchctl print "$LAUNCHD_DOMAIN/$WATCHDOG_LABEL" >/dev/null 2>&1; then
    break
  fi
  /bin/sleep 0.1
done
/bin/launchctl bootstrap "$LAUNCHD_DOMAIN" "$AGENT_PLIST"
/bin/launchctl bootstrap "$LAUNCHD_DOMAIN" "$WATCHDOG_PLIST"
/bin/launchctl enable "$LAUNCHD_DOMAIN/$AGENT_LABEL" 2>/dev/null || true
/bin/launchctl enable "$LAUNCHD_DOMAIN/$WATCHDOG_LABEL" 2>/dev/null || true
