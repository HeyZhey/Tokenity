# HTTP Node Agent Protocol

Tokenity uses HTTP for its cluster control plane. Installing Tokenity on each
Mac starts a launchd-managed Node Agent on port `9100`; users do not enable SSH,
exchange SSH keys, or provide machine passwords.

## Control plane and data plane

- **Control plane:** JSON over HTTP between TokenityControl and Node Agents,
  and between the coordinator Agent and worker Agents.
- **Data plane:** MLX ring TCP or JACCL/RDMA between ranks. Model tensors do not
  pass through the HTTP control API.
- **Client API:** the long-lived coordinator Node Agent is also the stable
  OpenAI gateway on port `9100`. It routes to isolated runtime ports only after
  instance quorum and the one-token readiness probe succeed.

## Lifecycle

1. TokenityControl sends `POST /v1/node/start-distributed-openai` to the
   coordinator Agent with each selected node's `agent_url`.
2. The coordinator starts rank 0 locally.
3. The coordinator sends a typed `POST /v1/node/start-distributed-rank` request
   to each worker Agent.
4. A worker uses its installed Python runtime and starts only the fixed
   `tokenity distributed-openai serve` entry point. It rejects a caller-selected
   executable and reports an error if the MLX data-plane connection is not made.
5. The runtime publishes atomic per-rank readiness, model revision, tokenizer,
   warmup, request timeline, and memory evidence. The coordinator exposes an
   instance as routable only when the planned rank quorum matches.
6. `POST /v1/node/instances/{instance_id}/stop` unloads only that process group.
   A request lease returns `409` while an inference request is still active.

## Relevant endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/node/info` | Runtime, network, RDMA, memory, and Agent details. |
| `GET` | `/v1/node/models` | Scan the configured local model root. |
| `GET` | `/v1/node/status` | Report locally supervised roles and failures. |
| `GET` | `/v1/node/instances` | List lifecycle, reservation, port, and usage state. |
| `GET` | `/v1/node/instances/{id}/quorum` | Verify all planned rank evidence. |
| `GET` | `/v1/gateway/routes` | List currently routable model instances. |
| `GET` | `/v1/models` | Stable gateway model catalog. |
| `POST` | `/v1/chat/completions` | Stable streaming/non-streaming gateway route. |
| `POST` | `/v1/video/generations` | Stable MiniMax H3 video route and SSE passthrough. |
| `POST` | `/v1/node/start-distributed-openai` | Start a cluster from its coordinator. |
| `POST` | `/v1/node/start-distributed-rank` | Internal typed worker-rank start. |
| `POST` | `/v1/node/start-minimax-h3-video` | Plan or start a native MiniMax H3 video instance. |
| `POST` | `/v1/node/start-minimax-h3-video-rank` | Version-gated internal TP2 worker-rank start. |
| `POST` | `/v1/node/instances/{id}/stop` | Stop one instance without touching siblings. |
| `POST` | `/v1/node/stop-role` | Stop a local role and fan out when applicable. |

`GET /v1/node/info` includes an `agent_contract` object. Contract version `1`
advertises stable capability names such as `managed_instances`,
`instance_quorum`, and `instance_runtimes`. Control clients must treat a missing
contract as a legacy Agent, while an Agent that advertises a capability but
omits its required telemetry is unhealthy.

It also includes `machine_id`, a one-way hashed platform identity. Unlike a LAN
address, this value remains stable across DHCP changes and lets TokenityControl
update a saved Mac in place. The raw platform UUID or serial number is never
returned.

## Local-network discovery

TokenityControl performs automatic active discovery when the app starts and
then every 30 seconds. It enumerates private/link-local IPv4 subnets, probes the
known Agent ports `9100` and `9200`, and accepts only responses that decode as
the typed `/v1/node/info` contract. Saved endpoints are always probed as seeds,
so discovery still works when an interface has an unusually broad netmask.

Results are coalesced by `machine_id`, with backward-compatible matching by
RDMA address, Node ID, and hostname/user for Agents deployed before
`machine_id`. Consequently a Mac that moves from `<node-a-lan-ip>` to
`<node-a-lan-ip>` keeps its selection, coordinator identity, models, and nickname.
When one physical Mac exposes both the stable Agent on `:9100` and an isolated
H3 Agent on `:9200`, the general cluster topology prefers `:9100`; Video retains
its independently configured H3 endpoint. Updated control endpoints are saved
locally for the next app launch. The **Cluster → Discover Macs** action runs the
same scan immediately.

The `minimax_h3_video` capability means the Agent can validate, supervise, and
proxy the native H3 backend and accept the typed TP2 worker endpoint.
Distributed launch additionally requires this capability on both Agents,
native backend protocol version `1`, and matching verified TP2 shards. An
unmodified upstream `mlx-serve` is accepted for single-Mac serving and is
rejected with HTTP `412` for a two-Mac request.

Both `/v1/node/info` and `/v1/node/status` retain the legacy singleton
`cluster_runtime` field and also publish `cluster_runtimes`, one entry per
`instance_id`. Multi-instance clients must select the entry matching their
active instance rather than using the last-started singleton.

## Example

```bash
curl -X POST http://<node-a-lan-ip>:9100/v1/node/start-distributed-openai \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "${TOKENITY_MODEL_ROOT}/Qwen3.5-122B-A10B-4bit",
    "connection_mode": "jaccl",
    "dry_run": false,
    "nodes": [
      {
        "id": "mac-a",
        "agent_url": "http://<node-a-lan-ip>:9100",
        "lan_ip": "<node-a-lan-ip>",
        "rdma_ip": "<node-a-rdma-ip>",
        "rdma_devices": ["rdma_en4"]
      },
      {
        "id": "mac-b",
        "agent_url": "http://<node-b-lan-ip>:9100",
        "lan_ip": "<node-b-lan-ip>",
        "rdma_ip": "<node-b-rdma-ip>",
        "rdma_devices": ["rdma_en5"]
      }
    ]
  }'
```

No SSH host, username, key, or password is present in the request. The response
contains `instance_id`, `operation_id`, the allocated private runtime URL, and
the launch plan. Clients should use `http://<coordinator>:9100/v1`; the private
runtime URL is for readiness diagnosis and the Control app's load probe.

## Security boundary

The Agent does not provide a general-purpose command or shell endpoint. Rank
requests are schema-validated and restricted to the Agent's installed Python
runtime. Legacy `ssh` fields and Agent URLs containing usernames or passwords
are rejected. The current Agent and inference API do not provide authentication
or TLS, so ports `9100` and `8000` must remain on a trusted local network.
Network authentication is a separate hardening item; it is not replaced by
collecting users' operating-system passwords.

On macOS, use the installer-managed launchd service for multi-Mac inference.
Ad-hoc terminal Python processes may be subject to Local Network privacy rules
and are intended only for development or single-Mac tests.
