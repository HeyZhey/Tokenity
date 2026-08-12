#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_PATH="$ROOT/packaging/runtime/runtime-lock.json"
MANIFEST_TOOL="$ROOT/scripts/tokenity-runtime-manifest.py"
DESTINATION="${TOKENITY_RUNTIME_CACHE:-$ROOT/dist/runtime-cache/TokenityRuntime}"
PRIMARY_SOURCE="${1:-}"
VERIFICATION_SOURCE="${2:-}"

if [[ -z "$PRIMARY_SOURCE" ]]; then
  echo "Usage: $0 <runtime-source> [verification-runtime-source]" >&2
  echo "A source may be local or rsync-style user@host:/path/to/TokenityRuntime." >&2
  exit 2
fi

TEMPORARY_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/tokenity-runtime-import.XXXXXX")"
cleanup() {
  case "$TEMPORARY_ROOT" in
    "${TMPDIR:-/tmp}"/tokenity-runtime-import.*)
      rm -rf "$TEMPORARY_ROOT"
      ;;
  esac
}
trap cleanup EXIT

copy_source() {
  local source="$1"
  local destination="$2"
  mkdir -p "$destination"
  rsync -a \
    --delete \
    --delete-excluded \
    --exclude ".DS_Store" \
    --exclude ".lock" \
    --exclude "CACHEDIR.TAG" \
    --exclude ".git" \
    --exclude "cache" \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude "*.pyo" \
    "${source%/}/" \
    "$destination/"
}

PRIMARY_STAGING="$TEMPORARY_ROOT/primary"
echo "Importing primary Tokenity Runtime..."
copy_source "$PRIMARY_SOURCE" "$PRIMARY_STAGING"
"$MANIFEST_TOOL" prepare "$PRIMARY_STAGING" \
  --lock "$LOCK_PATH" \
  --output "$PRIMARY_STAGING/runtime-manifest.json"

if [[ -n "$VERIFICATION_SOURCE" ]]; then
  VERIFICATION_STAGING="$TEMPORARY_ROOT/verification"
  echo "Importing verification Tokenity Runtime..."
  copy_source "$VERIFICATION_SOURCE" "$VERIFICATION_STAGING"
  "$MANIFEST_TOOL" prepare "$VERIFICATION_STAGING" \
    --lock "$LOCK_PATH" \
    --output "$VERIFICATION_STAGING/runtime-manifest.json"
  if ! cmp -s \
    "$PRIMARY_STAGING/runtime-manifest.json" \
    "$VERIFICATION_STAGING/runtime-manifest.json"; then
    echo "The normalized Runtime trees do not have the same identity." >&2
    diff -u \
      "$PRIMARY_STAGING/runtime-manifest.json" \
      "$VERIFICATION_STAGING/runtime-manifest.json" >&2 || true
    exit 1
  fi
  echo "Primary and verification Runtime identities match."
fi

mkdir -p "$DESTINATION"
rsync -a --delete "$PRIMARY_STAGING/" "$DESTINATION/"
"$MANIFEST_TOOL" verify "$DESTINATION" \
  --lock "$LOCK_PATH" \
  --manifest "$DESTINATION/runtime-manifest.json"

echo "Prepared Tokenity Runtime cache:"
echo "$DESTINATION"
sed -n \
  -e '/"runtime_id"/p' \
  -e '/"minimum_macos"/p' \
  -e '/"tree_sha256"/p' \
  "$DESTINATION/runtime-manifest.json"
