#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TOKENITY_TEST_PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

echo "Checking consolidated distributed backend..."
"$PYTHON" -c 'from tokenity.serving.distributed_openai import TokenityDistributedRuntime; assert TokenityDistributedRuntime.__name__ == "TokenityDistributedRuntime"'

echo "Checking installer scripts..."
bash -n "$ROOT/scripts/package-tokenity-dmg.sh"
bash -n "$ROOT/scripts/import-tokenity-runtime.sh"
"${PYTHON_BIN:-python3}" -m py_compile "$ROOT/scripts/tokenity-runtime-manifest.py"

echo "Running Python tests..."
"$PYTHON" -m pytest -q "$ROOT/tests"

echo "Running Swift tests..."
swift test --package-path "$ROOT/apps/TokenityControl"

echo "Building a fresh TokenityControl app bundle..."
APP_BUNDLE="$("$ROOT/scripts/build-tokenity-control-app.sh")"
APP_EXECUTABLE="$APP_BUNDLE/Contents/MacOS/TokenityControl"
test -x "$APP_EXECUTABLE"

for marker in \
  "The model returned reasoning but did not finish a final answer." \
  "ThinkingTagStreamParser" \
  "Tokenity, Distributed AI" \
  "Search conversations" \
  "WELCOME TO TOKENITY"; do
  if ! LC_ALL=C grep -aFq "$marker" "$APP_EXECUTABLE"; then
    echo "Built app does not contain the current Chat workspace marker: $marker" >&2
    exit 1
  fi
done

echo "Stable baseline verified: $APP_BUNDLE"
