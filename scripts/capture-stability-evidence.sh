#!/bin/zsh
set -eu

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 OUTPUT_DIRECTORY AGENT_URL [AGENT_URL ...]" >&2
  echo "Example: $0 ./evidence http://node-a.local:9100 http://node-b.local:9100" >&2
  exit 2
fi

OUTPUT_DIRECTORY="$1"
shift
/bin/mkdir -p "$OUTPUT_DIRECTORY"

capture_http() {
  local url="$1"
  local stem="$2"
  /usr/bin/curl --noproxy '*' --silent --show-error --max-time 10 \
    --output "$OUTPUT_DIRECTORY/$stem.body" \
    --write-out '%{http_code}\n' "$url" > "$OUTPUT_DIRECTORY/$stem.http" || true
}

node_index=0
for agent_url in "$@"; do
  node_index=$((node_index + 1))
  stem="node-$node_index"
  base_url="${agent_url%/}"
  capture_http "$base_url/v1/node/health" "$stem-health"
  capture_http "$base_url/v1/node/info" "$stem-info"
  capture_http "$base_url/v1/node/status" "$stem-status"
  capture_http "$base_url/v1/node/instances" "$stem-instances"
  capture_http "$base_url/v1/models" "$stem-models"
done

/usr/bin/shasum -a 256 "$OUTPUT_DIRECTORY"/*(.) > "$OUTPUT_DIRECTORY/SHA256SUMS"
echo "$OUTPUT_DIRECTORY"
