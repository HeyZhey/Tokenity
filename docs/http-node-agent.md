# HTTP Node Agent Protocol

Tokenity uses HTTP for its cluster control plane. Installing Tokenity on each
Mac starts a launchd-managed Node Agent on port `9100`; users do not enable SSH,
exchange SSH keys, or provide machine passwords.

## Control plane and data plane

- **Control plane:** JSON over HTTP between TokenityControl and Node Agents,
  and between the coordinator Agent and worker Agents.
- **Data plane:** MLX ring TCP or JACCL/RDMA between ranks. Model tensors do not
  pass through the HTTP control API.
- **Client API:** OpenAI-compatible HTTP on coordinator port `8000` after the
  model is ready.

## Lifecycle

1. TokenityControl sends `POST /v1/node/start-distributed-openai` to the
   coordinator Agent with each selected node's `agent_url`.
2. The coordinator starts rank 0 locally.
3. The coordinator sends a typed `POST /v1/node/start-distributed-rank` request
   to each worker Agent.
4. A worker uses its installed Python runtime and starts only the fixed
   `tokenity distributed-openai serve` entry point. It rejects a caller-selected
   executable and reports an error if the MLX data-plane connection is not made.
5. `POST /v1/node/stop-role` on the coordinator fans out the stop request to
   worker Agents before stopping rank 0.

The legacy `/v1/node/start-official-mlx-lm` endpoint returns HTTP `410` because
upstream `mlx.launch` uses SSH to create remote ranks.

## Relevant endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/node/info` | Runtime, network, RDMA, memory, and Agent details. |
| `GET` | `/v1/node/models` | Scan the configured local model root. |
| `GET` | `/v1/node/status` | Report locally supervised roles and failures. |
| `POST` | `/v1/node/start-distributed-openai` | Start a cluster from its coordinator. |
| `POST` | `/v1/node/start-distributed-rank` | Internal typed worker-rank start. |
| `POST` | `/v1/node/stop-role` | Stop a local role and fan out when applicable. |

## Example

```bash
curl -X POST http://192.168.5.23:9100/v1/node/start-distributed-openai \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/Users/Shared/TokenityModels/Qwen3.5-122B-A10B-4bit",
    "connection_mode": "jaccl",
    "dry_run": false,
    "nodes": [
      {
        "id": "mac-a",
        "agent_url": "http://192.168.5.23:9100",
        "lan_ip": "192.168.5.23",
        "rdma_ip": "192.168.0.1",
        "rdma_devices": ["rdma_en4"]
      },
      {
        "id": "mac-b",
        "agent_url": "http://192.168.5.75:9100",
        "lan_ip": "192.168.5.75",
        "rdma_ip": "192.168.0.2",
        "rdma_devices": ["rdma_en5"]
      }
    ]
  }'
```

No SSH host, username, key, or password is present in the request.

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
