#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${1:-$ROOT/dist/runtime-cache/TokenityRuntime}"
LOCK="$ROOT/packaging/runtime/runtime-lock.json"
PYTHON="$RUNTIME/pythons/cpython-3.12-macos-aarch64-none/bin/python3.12"
BACKEND="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["vlm_backend"]["backend_name"])' "$LOCK")"
VENV="$RUNTIME/backends/$BACKEND/.venv"
UV="${TOKENITY_UV:-uv}"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$UV" venv --python "$PYTHON" "$VENV"
fi
"$UV" pip sync --python "$VENV/bin/python" "$ROOT/packaging/runtime/vlm-requirements.txt"
echo "Prepared isolated MLX-VLM runtime: $VENV"
