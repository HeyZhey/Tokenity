#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${TOKENITY_VERSION:-0.1.0}"
DIST_DIR="$ROOT/dist"
WORK_DIR="$DIST_DIR/agent-update-work"
PAYLOAD_DIR="$WORK_DIR/payload"
SCRIPTS_DIR="$WORK_DIR/scripts"
PKG_PATH="$DIST_DIR/Tokenity-NodeAgent-${VERSION}.pkg"
INSTALLED_RUNTIME="/Users/Shared/TokenityRuntime/current/.venv/bin/python"
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
/bin/mkdir -p "$PAYLOAD_DIR/Users/Shared/TokenityCode" "$SCRIPTS_DIR" "$DIST_DIR"

/usr/bin/rsync -a \
  --delete \
  --exclude '.DS_Store' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$ROOT/tokenity/" \
  "$PAYLOAD_DIR/Users/Shared/TokenityCode/tokenity/"

{
  /usr/bin/printf '#!/bin/zsh\nset -eu\n'
  /usr/bin/printf 'EXPECTED_MLX="%s"\n' "$EXPECTED_MLX"
  /usr/bin/printf 'EXPECTED_MLX_LM="%s"\n' "$EXPECTED_MLX_LM"
  /bin/cat <<'SCRIPT'

PYTHON="/Users/Shared/TokenityRuntime/current/.venv/bin/python"
PLIST="/Library/LaunchDaemons/ai.tokenity.node-agent.plist"
AGENT_URL="http://127.0.0.1:9100"

remove_legacy_user_agents() {
  local legacy_plist legacy_uid
  for legacy_plist in /Users/*/Library/LaunchAgents/local.tokenity.node-agent.plist(N); do
    legacy_uid="$(/usr/bin/stat -f '%u' "$legacy_plist" 2>/dev/null || true)"
    if [[ -n "$legacy_uid" ]]; then
      /bin/launchctl bootout \
        "gui/$legacy_uid/local.tokenity.node-agent" 2>/dev/null || \
        /bin/launchctl bootout "gui/$legacy_uid" "$legacy_plist" 2>/dev/null || true
    fi
    /bin/rm -f "$legacy_plist"
  done
}

if [[ ! -x "$PYTHON" ]]; then
  echo "Tokenity Runtime is missing at $PYTHON; refusing a code-only Agent update." >&2
  exit 1
fi
if [[ ! -f "$PLIST" ]]; then
  echo "The installer-managed Node Agent LaunchDaemon is missing at $PLIST." >&2
  exit 1
fi

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

/bin/chmod -R a+rX /Users/Shared/TokenityCode
/usr/bin/find /Users/Shared/TokenityCode/tokenity -type d -name __pycache__ -prune -exec /bin/rm -rf {} + 2>/dev/null || true

/bin/launchctl kickstart -k "system/\$LABEL"

for _ in {1..30}; do
  payload="\$(/usr/bin/curl --noproxy '*' --silent --max-time 2 "\$AGENT_URL/v1/node/info" 2>/dev/null || true)"
  if [[ "\$payload" == *"managed_instances"* ]] &&
     [[ "\$payload" == *"\$EXPECTED_REVISION"* ]]; then
    echo "Tokenity Node Agent updated to \$EXPECTED_REVISION."
    exit 0
  fi
  /bin/sleep 1
done

echo "The updated Node Agent did not publish the expected managed-instance contract." >&2
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
