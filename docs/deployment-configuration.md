# Deployment configuration

Tokenity does not embed a deployment's IP addresses, usernames, checkout
paths, SSH keys, model locations, or Thunderbolt topology. Configure those
values at deployment time through the UI or environment.

## Filesystem layout

`TOKENITY_DATA_ROOT` changes the complete layout. Individual variables take
precedence when set:

| Variable | Purpose | Derived default |
| --- | --- | --- |
| `TOKENITY_DATA_ROOT` | Common Tokenity data root | Packaged macOS layout from `install-layout.env`; platform user-data directory for source runs |
| `TOKENITY_CODE_ROOT` | Installed Python package tree | `$TOKENITY_DATA_ROOT/Code` |
| `TOKENITY_MODEL_ROOT` | Model inventory | `$TOKENITY_DATA_ROOT/Models` |
| `TOKENITY_RUNTIME_ROOT` | Pinned Python/MLX runtime | `$TOKENITY_DATA_ROOT/Runtime` |
| `TOKENITY_RUNTIME_PYTHON` | Exact Python executable | `$TOKENITY_RUNTIME_ROOT/current/.venv/bin/python` |
| `TOKENITY_STATE_ROOT` | Instance and watchdog state | `$TOKENITY_DATA_ROOT/State` |
| `TOKENITY_LOG_ROOT` | Agent and watchdog logs | `$TOKENITY_DATA_ROOT/Logs` |

Offline Runtime installers are discovered through macOS's mounted-volume API,
without assuming a filesystem mount root. The macOS package build reads
`packaging/install-layout.env`. Override
`TOKENITY_INSTALL_ROOT` when building a package for a different managed
layout. The generated launchd jobs receive all resolved paths explicitly, and
the resolved layout is embedded in the control app's deployment configuration.
Package install roots must be absolute and cannot contain whitespace because
Python entrypoint shebangs must remain directly executable.

## Node endpoints

| Variable | Purpose |
| --- | --- |
| `TOKENITY_AGENT_URL` | Default local/control Node Agent |
| `TOKENITY_NODE_AGENT_URLS` | Comma-separated discovery seed URLs |
| `TOKENITY_NODE_AGENT_PORT` | Fallback Agent port when a typed node only supplies a LAN address |
| `TOKENITY_H3_COORDINATOR_AGENT` | MiniMax H3 coordinator Agent |
| `TOKENITY_H3_WORKER_AGENT` | MiniMax H3 worker Agent |

Prefer stable DNS or mDNS names. With no explicit configuration, the app seeds
only the local Agent at `127.0.0.1:9100`; remote Mac slots contain no endpoint,
so a new checkout cannot contact a previous developer's LAN or mistake a local
port for another Mac. Automatic LAN discovery claims an unbound slot only after
`/v1/node/info` verifies the Agent. The **Connect by URL** action performs the
same verification when subnet scanning is unavailable.

If several unbound remote Macs answer the first scan, an Agent with an active
RDMA link is assigned to the default worker slot first. Additional verified
Macs remain available but are not selected into the cluster automatically.

Verified endpoints are persisted by stable node identity. On upgrade, obsolete
remote `127.0.0.1:9200` and `127.0.0.1:9300` placeholder values are discarded
instead of being treated as real Macs. Values supplied by
`TOKENITY_NODE_AGENT_URLS` take precedence over persisted discovery state.

## MiniMax H3 and RDMA

| Variable | Purpose |
| --- | --- |
| `TOKENITY_H3_MODEL_PATH` | H3 model root |
| `TOKENITY_H3_BINARY_PATH` | Native `mlx-serve` executable |
| `TOKENITY_RDMA_CIDRS` | Optional comma-separated RDMA address ranges |
| `TOKENITY_TB_INTERFACE` | Optional packaged Thunderbolt interface |
| `TOKENITY_TB_LOCAL_IP` | Optional packaged Thunderbolt local address |
| `TOKENITY_TB_PEER_IP` | Optional packaged Thunderbolt peer address |

The three `TOKENITY_TB_*` values must be supplied together. Without them, the
installer does not alter a network interface or install a keepalive job. When
`TOKENITY_RDMA_CIDRS` is absent, the runtime accepts a private or link-local
address found on the detected RDMA interface rather than assuming one subnet.

## Memory admission

**Settings → Advanced → Memory Admission** controls the fixed memory headroom
used for the next language or video model load:

| Policy | Fixed headroom | Intended use |
| --- | ---: | --- |
| Safe | 25% | Default and previous Tokenity behavior |
| Balanced | 15% | Recommended for 256 GiB and 512 GiB Macs |
| Aggressive | 10% | Larger checkpoints when the workload is controlled |
| Custom | 5%–40% | Explicit deployment-specific headroom |
| Disabled | 0% | Development builds only; prominently warned in the UI |

The setting changes only the fixed headroom. It never removes the live
non-reclaimable-memory observation, reservations owned by other Tokenity model
instances, or the macOS memory-pressure boundary. Even the development-only
Disabled policy remains bounded by those three signals.

For a requested headroom ratio `h`, the reservation ledger first limits usable
memory to `total × (1 − h)` and subtracts other instance reservations. The live
system bound independently subtracts current in-use memory and applies the
memory-pressure available ratio. Admission uses the lower of the ledger and
live-system bounds, so lowering `h` cannot make existing allocations or system
pressure invisible.

The selected ratio is carried to every rank and retained with the model
instance for post-load reservation reconciliation and process recovery. Safe is
the protocol default, allowing the app to omit the new field when talking to an
older Agent; a non-Safe policy requires an updated Agent on every selected Mac.

## Preflight and evidence capture

The preflight command requires the complete topology instead of shipping a
machine-specific default:

```bash
./scripts/preflight-tokenity-cluster.py \
  --node 'node-a,http://node-a.local:9100,<node-a-lan>,<node-a-rdma-device>,<node-a-rdma-ip>' \
  --node 'node-b,http://node-b.local:9100,<node-b-lan>,<node-b-rdma-device>,<node-b-rdma-ip>' \
  --model-path "$TOKENITY_MODEL_ROOT/<model-directory>" \
  --python-path "$TOKENITY_RUNTIME_PYTHON"
```

Evidence capture is HTTP-only and accepts any number of explicit Agent URLs:

```bash
./scripts/capture-stability-evidence.sh ./evidence \
  http://node-a.local:9100 http://node-b.local:9100
```
