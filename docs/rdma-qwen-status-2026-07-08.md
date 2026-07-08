# Tokenity RDMA + Qwen Status - 2026-07-08

This is the known-good local baseline after rebooting Mac A and Mac B.

## Repository

- UI/control project: `/Users/zxc/Documents/Tokenity`
- Active app bundle: `/Users/zxc/Documents/Tokenity/apps/TokenityControl/.build/arm64-apple-macosx/debug/TokenityControl.app`
- Legacy UI path to avoid: `/Users/zxc/Documents/MLX-Distributed/apps/TokenityControl`
- `/Users/zxc/Documents/MLX-Distributed` is not currently a git repository.
- No references to old Mac B IP `192.168.5.61` were found.

## Machines

- Mac A: `apple@192.168.5.23`
- Mac B: `probriefing@192.168.5.75`
- Runtime Python: `/Users/Shared/TokenityRuntime/current/.venv/bin/python`
- Model: `/Users/Shared/TokenityModels/Qwen3.5-122B-A10B-4bit`
- Versions confirmed on both Macs:
  - `mlx 0.31.2`
  - `mlx-lm 0.31.3`

## Node Agents

- Mac B node-agent came up after reboot on port `9100`.
- Mac A node-agent did not auto-start after reboot and was started manually.
- Startup command used for Mac A:

```bash
ssh apple@192.168.5.23 'mkdir -p "$HOME/Library/Application Support/Tokenity/logs"; nohup env PATH=/Users/Shared/TokenityRuntime/current/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin PYTHONPATH=/Users/Shared/TokenityCode /Users/Shared/TokenityRuntime/current/.venv/bin/python -u -m tokenity node-agent --host 0.0.0.0 --port 9100 > "$HOME/Library/Application Support/Tokenity/logs/node-agent.log" 2>&1 &'
```

## Thunderbolt RDMA Baseline

- Mac A: `en4`, `rdma_en4`, `192.168.0.1/30`
- Mac B: `en5`, `rdma_en5`, `192.168.0.2/30`
- After reboot both sides reported:
  - `ifconfig` status: `active`
  - `ibv_devinfo` state: `PORT_ACTIVE`
- A to B Thunderbolt ping: `10/10` packets received, `0.0%` loss, about `0.44 ms` average.
- A to B BatchMode SSH over Thunderbolt passed:
  - `ssh probriefing@192.168.0.2 hostname`

## JACCL Smoke Test

The direct MLX/JACCL smoke test passed with:

```text
rank=0 size=2 all_sum=3.0
rank=1 size=2 all_sum=3.0
```

Hostfile shape:

```json
[
  {"ssh":"127.0.0.1","ips":["192.168.0.1"],"rdma":[null,"rdma_en4"]},
  {"ssh":"probriefing@192.168.0.2","ips":[],"rdma":["rdma_en5",null]}
]
```

## Qwen Load Baseline

NodeAgent API was used to start Tokenity distributed OpenAI with:

- `connection_mode`: `jaccl`
- `starting_port`: `30020`
- OpenAI port: `8000`
- Coordinator: `192.168.0.1`

The model loaded successfully over RDMA/JACCL.

Verified responses on Mac A:

```text
/v1/readiness -> phase=ready, world_size=2
/v1/models    -> HTTP 200, Qwen3.5-122B-A10B-4bit
```

Final readiness payload:

```json
{
  "phase": "ready",
  "rank": 0,
  "world_size": 2,
  "backend": "mlx-distributed",
  "model": "/Users/Shared/TokenityModels/Qwen3.5-122B-A10B-4bit",
  "message": "Runtime is accepting generation requests."
}
```

A light `/v1/chat/completions` request returned a valid OpenAI-compatible response.

Cold load time in this run was roughly `2.5-3 minutes`, reaching chunk `2107/2107` before readiness became `ready`.

## Current Runtime State At Capture

- RDMA model service was intentionally left running.
- A node-agent role `distributed-openai` was running.
- Mac B rank process was running.
- No orphan process was detected beyond the active model/rank set.
- Stop command:

```bash
ssh apple@192.168.5.23 'curl --noproxy "*" -sS -X POST http://127.0.0.1:9100/v1/node/stop-role -H "Content-Type: application/json" -d "{\"role\":\"distributed-openai\",\"timeout\":10}"'
```

## UI Fixes Included In This Baseline

- Tokenity Control now refreshes `/v1/node/info` during selected-node status refresh, so RDMA status is not stuck on sample data.
- RDMA model Load now refreshes node info first and blocks early when `rdma_enabled` is false.
- NodeAgent `rdma_errors` are surfaced in the UI readiness/load message.
- Standard Network load path remains unaffected.

