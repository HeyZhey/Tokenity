#!/bin/zsh
set -eu

if [[ "$EUID" != "0" ]]; then
  echo "Run this uninstaller with sudo." >&2
  exit 1
fi

AGENT_PLIST="/Library/LaunchDaemons/ai.tokenity.node-agent.plist"
WATCHDOG_PLIST="/Library/LaunchDaemons/ai.tokenity.node-agent-watchdog.plist"
TB_PLIST="/Library/LaunchDaemons/ai.tokenity.thunderbolt-keepalive.plist"
TB_SCRIPT="/usr/local/bin/tokenity-tb-keepalive"
AGENT_URL="http://127.0.0.1:9100"
plist_value() {
  /usr/libexec/PlistBuddy -c "Print :EnvironmentVariables:$1" "$AGENT_PLIST" 2>/dev/null || true
}
INSTALL_ROOT="${TOKENITY_DATA_ROOT:-$(plist_value TOKENITY_DATA_ROOT)}"
STATE_ROOT="${TOKENITY_STATE_ROOT:-$(plist_value TOKENITY_STATE_ROOT)}"
LOG_ROOT="${TOKENITY_LOG_ROOT:-$(plist_value TOKENITY_LOG_ROOT)}"
MODEL_ROOT="${TOKENITY_MODEL_ROOT:-$(plist_value TOKENITY_MODEL_ROOT)}"
if [[ -z "$INSTALL_ROOT" || -z "$STATE_ROOT" || -z "$LOG_ROOT" ]]; then
  echo "Cannot determine the Tokenity install layout; set TOKENITY_DATA_ROOT, TOKENITY_STATE_ROOT, and TOKENITY_LOG_ROOT." >&2
  exit 1
fi
MAINTENANCE_PATH="$STATE_ROOT/maintenance.json"

/bin/mkdir -p "$STATE_ROOT"
/usr/bin/printf '{"until":%s,"reason":"uninstall"}\n' \
  "$(( $(/bin/date +%s) + 600 ))" > "$MAINTENANCE_PATH"
/bin/launchctl bootout system "$WATCHDOG_PLIST" 2>/dev/null || true
/bin/launchctl bootout system "$TB_PLIST" 2>/dev/null || true

if /usr/bin/curl --noproxy '*' --silent --fail --max-time 3 \
  "$AGENT_URL/v1/node/health" >/dev/null; then
  /usr/bin/curl --noproxy '*' --silent --show-error --max-time 5 \
    -H 'Content-Type: application/json' -d '{}' \
    "$AGENT_URL/v1/node/request-stop-all" >/dev/null 2>&1 || true
  /usr/bin/curl --noproxy '*' --silent --show-error --max-time 40 \
    -H 'Content-Type: application/json' -d '{"timeout":30}' \
    "$AGENT_URL/v1/node/stop-all" >/dev/null 2>&1 || true
fi

if /usr/bin/pgrep -f 'tokenity distributed-openai serve' >/dev/null 2>&1; then
  /bin/rm -f "$MAINTENANCE_PATH"
  /bin/launchctl bootstrap system "$WATCHDOG_PLIST" 2>/dev/null || true
  echo "A model runtime is still active. The uninstaller left the Agent installed so it cannot become orphaned." >&2
  exit 1
fi

/bin/launchctl bootout system "$AGENT_PLIST" 2>/dev/null || true
/bin/rm -f "$WATCHDOG_PLIST" "$AGENT_PLIST" "$TB_PLIST" "$TB_SCRIPT"
/bin/rm -f \
  "$STATE_ROOT/node-agent-watchdog.json" \
  "$STATE_ROOT/maintenance.json" \
  "$STATE_ROOT/node-agent-watchdog.lock" \
  "$STATE_ROOT/rdma-reset-request"
/bin/rm -f \
  "$LOG_ROOT/node-agent-watchdog.stdout.log" \
  "$LOG_ROOT/node-agent-watchdog.stderr.log"

while IFS= read -r home_directory; do
  for legacy_label in local.tokenity.node-agent dev.tokenity.dns-sd; do
    legacy_plist="$home_directory/Library/LaunchAgents/$legacy_label.plist"
    [[ -f "$legacy_plist" ]] || continue
    legacy_uid="$(/usr/bin/stat -f '%u' "$legacy_plist" 2>/dev/null || true)"
    if [[ -n "$legacy_uid" ]]; then
      /bin/launchctl bootout "gui/$legacy_uid/$legacy_label" 2>/dev/null || true
    fi
    /bin/rm -f "$legacy_plist"
  done
done < <(/usr/bin/dscacheutil -q user | /usr/bin/awk '/^dir: / {sub(/^dir: /, ""); print}')

echo "Removed the Tokenity Node Agent, watchdog, and Thunderbolt keepalive jobs. Model files in ${MODEL_ROOT:-$INSTALL_ROOT/Models} were preserved."
