#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TOKENITY_TEST_PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

echo "Checking consolidated distributed backend..."
"$PYTHON" -c 'from tokenity.serving.distributed_openai import TokenityDistributedRuntime; assert TokenityDistributedRuntime.__name__ == "TokenityDistributedRuntime"'

echo "Running Python tests..."
"$PYTHON" -m pytest -q "$ROOT/tests"

echo "Running Swift tests..."
swift test --package-path "$ROOT/apps/TokenityControl"

echo "Building a fresh TokenityControl app bundle..."
APP_BUNDLE="$("$ROOT/scripts/build-tokenity-control-app.sh")"
APP_EXECUTABLE="$APP_BUNDLE/Contents/MacOS/TokenityControl"
test -x "$APP_EXECUTABLE"

if ! strings "$APP_EXECUTABLE" | grep -Fq "The model returned reasoning but did not finish a final answer."; then
  echo "Built app does not contain the current chat-streaming baseline." >&2
  exit 1
fi

echo "Stable baseline verified: $APP_BUNDLE"
