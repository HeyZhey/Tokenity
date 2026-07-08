# Tokenity Current Usage - RDMA + Qwen

Last verified: 2026-07-08

This document records the current working way to run Tokenity with Mac A + Mac B and Qwen3.5-122B-A10B-4bit.

## Important Paths

- Main Tokenity repo and UI project: `/Users/zxc/Documents/Tokenity`
- UI source: `/Users/zxc/Documents/Tokenity/apps/TokenityControl`
- UI launch script: `/Users/zxc/Documents/Tokenity/scripts/run-tokenity-control-app.sh`
- UI app bundle: `/Users/zxc/Documents/Tokenity/apps/TokenityControl/.build/arm64-apple-macosx/debug/TokenityControl.app`
- Do not use old UI path: `/Users/zxc/Documents/MLX-Distributed/apps/TokenityControl`
- Runtime Python: `/Users/Shared/TokenityRuntime/current/.venv/bin/python`
- Model path: `/Users/Shared/TokenityModels/Qwen3.5-122B-A10B-4bit`

## Machines

- Mac A: `apple@192.168.5.23`
- Mac B: `probriefing@192.168.5.75`
- Do not reintroduce the retired Mac B LAN address.

Thunderbolt RDMA layout:

- Mac A: `en4`, `rdma_en4`, `192.168.0.1/30`
- Mac B: `en5`, `rdma_en5`, `192.168.0.2/30`

## After Reboot

1. Check LAN SSH:

```bash
ssh apple@192.168.5.23 hostname
ssh probriefing@192.168.5.75 hostname
```

2. Start NodeAgent on Mac A if it is not already running:

```bash
ssh apple@192.168.5.23 'mkdir -p "$HOME/Library/Application Support/Tokenity/logs"; nohup env PATH=/Users/Shared/TokenityRuntime/current/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin PYTHONPATH=/Users/Shared/TokenityCode /Users/Shared/TokenityRuntime/current/.venv/bin/python -u -m tokenity node-agent --host 0.0.0.0 --port 9100 > "$HOME/Library/Application Support/Tokenity/logs/node-agent.log" 2>&1 &'
```

3. Start NodeAgent on Mac B if it is not already running:

```bash
ssh probriefing@192.168.5.75 'mkdir -p "$HOME/Library/Application Support/Tokenity/logs"; nohup env PATH=/Users/Shared/TokenityRuntime/current/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin PYTHONPATH=/Users/Shared/TokenityCode /Users/Shared/TokenityRuntime/current/.venv/bin/python -u -m tokenity node-agent --host 0.0.0.0 --port 9100 > "$HOME/Library/Application Support/Tokenity/logs/node-agent.log" 2>&1 &'
```

4. Launch the correct UI:

```bash
cd /Users/zxc/Documents/Tokenity
./scripts/run-tokenity-control-app.sh
```

## RDMA Preflight

Run these checks before using Thunderbolt RDMA:

```bash
ssh apple@192.168.5.23 'ifconfig en4 | egrep "inet |status:"; ibv_devinfo -d rdma_en4 2>/dev/null | egrep "hca_id:|state:"; ping -c 4 -W 1000 192.168.0.2'
ssh probriefing@192.168.5.75 'ifconfig en5 | egrep "inet |status:"; ibv_devinfo -d rdma_en5 2>/dev/null | egrep "hca_id:|state:"; ping -c 4 -W 1000 192.168.0.1'
ssh apple@192.168.5.23 'ssh -o BatchMode=yes -o ConnectTimeout=5 probriefing@192.168.0.2 hostname'
```

Expected:

- A `en4` status is `active`
- A `rdma_en4` state is `PORT_ACTIVE`
- B `en5` status is `active`
- B `rdma_en5` state is `PORT_ACTIVE`
- Thunderbolt ping has `0.0%` packet loss after any first-packet ARP warmup
- A can SSH to `probriefing@192.168.0.2`

If A shows `rdma_enabled:false`, `PORT_DOWN`, or Thunderbolt ports show no connected device, unplug and replug the Thunderbolt cable, then re-run the preflight.

## UI Workflow

1. Open Tokenity Control with the launch script above.
2. Go to `Cluster`.
3. Select both Mac A and Mac B.
4. Choose backend `Tokenity Distributed Server`.
5. Choose connection:
   - `Thunderbolt RDMA` for JACCL/RDMA.
   - `Standard Network` for the stable non-RDMA path.
6. Click `Create Cluster`.
7. Go to `Models`.
8. Click `Load` for `Qwen3.5-122B-A10B-4bit`.
9. Wait for the row to become `Loaded`.

Expected cold load time:

- Thunderbolt RDMA: about `2.5-3 minutes`
- Standard Network: about `2-3 minutes`

During load, `/v1/readiness` may show `loading_model` and `/v1/models` may return 503. That is expected until chunk `2107/2107` finishes.

## API Verification

Check readiness from Mac A:

```bash
ssh apple@192.168.5.23 'curl --noproxy "*" -sS http://127.0.0.1:8000/v1/readiness'
```

Expected:

```json
{"phase":"ready","world_size":2}
```

Check models:

```bash
ssh apple@192.168.5.23 'curl --noproxy "*" -sS http://127.0.0.1:8000/v1/models'
```

Expected:

- HTTP 200
- model id `Qwen3.5-122B-A10B-4bit`
- `tokenity.world_size` is `2`

Light chat check:

```bash
ssh apple@192.168.5.23 'curl --noproxy "*" -sS -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen3.5-122B-A10B-4bit\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":4}"'
```

## Direct JACCL Smoke Test

Use this when the UI says RDMA is blocked or before debugging model load:

```bash
ssh apple@192.168.5.23 'hostfile=$(mktemp /tmp/tokenity-rdma-smoke.XXXXXX.json); /usr/bin/printf "%s\n" '"'"'[{"ssh":"127.0.0.1","ips":["192.168.0.1"],"rdma":[null,"rdma_en4"]},{"ssh":"probriefing@192.168.0.2","ips":[],"rdma":["rdma_en5",null]}]'"'"' > "$hostfile"; /Users/Shared/TokenityRuntime/current/.venv/bin/mlx.launch --hostfile "$hostfile" --backend jaccl --starting-port 30020 --env PATH=/Users/Shared/TokenityRuntime/current/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin --env PYTHONPATH=/Users/Shared/TokenityCode --cwd /Users/Shared/TokenityCode --no-verify-script -- python -c '"'"'import mlx.core as mx; group=mx.distributed.init(); value=mx.array(float(group.rank()+1)); total=mx.distributed.all_sum(value, stream=mx.cpu); mx.eval(total); print(f"rank={group.rank()} size={group.size()} all_sum={float(total.item())}")'"'"'; rc=$?; rm -f "$hostfile"; exit $rc'
```

Expected:

```text
rank=0 size=2 all_sum=3.0
rank=1 size=2 all_sum=3.0
```

## Stop And Cleanup

Stop the active distributed model role:

```bash
ssh apple@192.168.5.23 'curl --noproxy "*" -sS -X POST http://127.0.0.1:9100/v1/node/stop-role -H "Content-Type: application/json" -d "{\"role\":\"distributed-openai\",\"timeout\":10}"'
```

Check for leftovers:

```bash
ssh apple@192.168.5.23 'ps auxww | egrep "mlx\\.launch|tokenity distributed-openai|Qwen3\\.5|ssh -tt" | grep -v egrep || true; lsof -nP -iTCP:8000 -sTCP:LISTEN 2>/dev/null || true'
ssh probriefing@192.168.5.75 'ps auxww | egrep "mlx\\.launch|tokenity distributed-openai|Qwen3\\.5|ssh -tt" | grep -v egrep || true; lsof -nP -iTCP:8000 -sTCP:LISTEN 2>/dev/null || true'
```

Only NodeAgent should remain on port `9100`.

## Troubleshooting Notes

- If Standard Network loads but Thunderbolt RDMA does not, first check RDMA preflight. Do not assume it is a model issue.
- If UI immediately blocks RDMA load, check the readiness message. The UI now refreshes `/v1/node/info` and surfaces NodeAgent `rdma_errors`.
- If UI shows old state after API-side loading, refresh/reopen the UI and check `/v1/readiness` directly.
- Local shell proxy variables can interfere with `curl 192.168.*`; use `curl --noproxy "*"`.
- High memory shown in UI after stop may include macOS cache. Confirm with process checks and `memory_pressure`.
