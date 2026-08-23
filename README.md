<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/TokenityControl/Sources/TokenityControl/Resources/TokenityBrandLockupDark.png">
    <source media="(prefers-color-scheme: light)" srcset="apps/TokenityControl/Sources/TokenityControl/Resources/TokenityBrandLockup.png">
    <img alt="Tokenity" src="apps/TokenityControl/Sources/TokenityControl/Resources/TokenityBrandLockup.png" width="220">
  </picture>
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

# Tokenity

Tokenity turns Apple-silicon Macs on a local network into a managed AI cluster.
It provides one native control plane for language and video inference, with
distributed execution, resource isolation, acceleration, and recovery built
into the runtime.

The project contains:

- **TokenityControl** — a native SwiftUI app for cluster discovery, topology,
  model management, chat, video generation, health, logs, and recovery.
- **Tokenity Node Agent** — a supervised HTTP service on each inference Mac.
- **Tokenity runtimes** — distributed MLX inference and workload-specific
  backends behind one OpenAI-compatible gateway.

## Product highlights

- **Mac cluster control plane** — discover, select, inspect, and operate
  Apple-silicon nodes from one native macOS application.
- **Automatic LAN discovery** — finds Node Agents on private networks, tracks
  Macs by stable `machine_id`, and repairs changed addresses without turning a
  remote node into loopback.
- **Fast distributed data plane** — uses Thunderbolt RDMA/JACCL when available,
  with readiness checks and standard-network modes for other deployments.
- **Accelerated inference** — metadata-driven model compatibility, Native MTP
  capability detection, and guarded activation instead of model-name switches.
- **Unified workloads** — OpenAI-compatible streaming and non-streaming LLM
  inference plus native video generation; MiniMax H3 is the currently validated
  video backend, not the product boundary.
- **Production lifecycle** — multiple isolated instances, automatic routing,
  request queues, timeouts, cancellation, memory admission, resource ledgers,
  health/readiness probes, watchdogs, and restart recovery.
- **Cluster-safe operations** — typed HTTP orchestration, coordinated rank
  startup and shutdown, and no SSH dependency in the product path.

## Architecture

```text
                    ┌────────── HTTP control plane ──────────┐
TokenityControl ───►│ Coordinator Agent      Worker Agents   │
LAN discovery ─────►│ health · routing       local watchdogs │
                    └──────────────┬─────────────────────────┘
                                   │ stable gateway :9100
OpenAI clients ─────────────────────┤
                                   ▼
                         LLM · video workloads
                          ╲                      ╱
                           Thunderbolt RDMA/JACCL
                              or standard network
```

Node Agents start and supervise only local processes. Distributed ranks are
coordinated through typed HTTP requests while model collectives use the chosen
data plane. SSH is not used by the product. Current hardware validation covers
single-node and two-node workloads; the control plane is agent-based rather
than a fixed two-machine topology.

## Requirements

- Apple-silicon Macs on the same trusted network.
- macOS 26.2+ and Swift 5.9+ for Tokenity app development.
- Python 3.10+ for backend development.
- A compatible MLX/MLX-LM runtime on every inference Mac.
- The same model path on every participating Mac for a distributed workload.
- TCP port `9100` reachable between TokenityControl and each Node Agent.
- For RDMA: an active direct Thunderbolt link, RDMA devices, and peer IPs.

The currently validated packaged runtime is Apple-silicon only and requires
macOS 26.2 or newer. Model weights are not included.

## Development setup

```bash
git clone <repository-url> Tokenity
cd Tokenity

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Build and open the native app:

```bash
./scripts/run-tokenity-control-app.sh
```

Run a development Node Agent:

```bash
tokenity node-agent --host 0.0.0.0 --port 9100
```

For persistent deployment, install the Node Agent package on every inference
Mac as described in [Installer and packaging](docs/installer-dmg.md).

## Basic workflow

1. Install the Node Agent on each Mac and open **Cluster**.
2. Let LAN discovery find the nodes, or connect to an Agent directly.
3. Select the Macs for the workload and confirm the data plane reports
   **Ready**.
4. Create the cluster topology.
5. Open **Models**, enter the shared absolute model directory, and select
   **Scan Models**.
6. Load a model, then use **Chat** or the external API.
7. Open **Video** for video-runtime checks, generation, streamed progress,
   cancellation, preview, and export.

The LLM model directory and H3 runtime paths are separate settings. A model is
listed only when it exists under the scanned directory on the selected Macs.

## OpenAI-compatible API

The coordinator exposes the stable gateway at:

```text
http://<coordinator-host>:9100/v1
```

Main endpoints:

- `GET /v1/models`
- `GET /v1/gateway/routes`
- `POST /v1/chat/completions`
- `POST /v1/video/generations`

Example:

```bash
export TOKENITY_HOST=<coordinator-host>

curl "http://${TOKENITY_HOST}:9100/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<model-id>",
    "messages": [{"role": "user", "content": "Hello from Tokenity"}],
    "stream": false
  }'
```

Authentication and TLS are not enabled by default. Keep the API on a trusted
local network.

## Validation

Run the complete local baseline:

```bash
./scripts/verify-stable-baseline.sh
```

This runs the Python tests, Swift tests, and a fresh TokenityControl debug app
build. Real distributed model, video, MTP, and RDMA validation additionally
requires the matching runtimes, checkpoints, and cluster hardware. The current
hardware baseline includes GLM 5.2 and MiniMax H3 TP2.

## Packaging

```bash
./scripts/build-tokenity-control-app.sh
./scripts/package-tokenity-dmg.sh
```

Release packaging, runtime pinning, signing status, and Node Agent installation
are documented in [Installer and packaging](docs/installer-dmg.md).

## Documentation

- [Deployment configuration](docs/deployment-configuration.md)
- [HTTP Node Agent](docs/http-node-agent.md)
- [External API](docs/external-api.md)
- [GLM 5.2 compatibility](docs/glm-5.2-compat.md)
- [MiniMax H3 video](docs/minimax-h3-video.md)
- [Native MTP](docs/native-mtp.md)
- [RDMA operations](docs/current-usage-rdma-qwen.md)
- [Stable baseline](docs/stable-baseline.md)

## Repository layout

```text
apps/TokenityControl/  SwiftUI application and tests
tokenity/              Python CLI, Node Agent, routing, and runtimes
tests/                 Python regression tests
scripts/               Build, verification, deployment, and packaging tools
docs/                  Focused operator and compatibility guides
```

## License

No open-source license has been declared for Tokenity. All rights are reserved.
See [Third-party notices](THIRD_PARTY_NOTICES.md) for separately licensed code.
