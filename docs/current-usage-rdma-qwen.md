# Tokenity Current Usage — HTTP + RDMA + Qwen

Updated: 2026-07-12

Native MTP remains off by default. See [Tokenity Native MTP MVP](native-mtp.md)
for the `off`/`auto`/`required` contract. The current
`Qwen3.5-122B-A10B-4bit` checkpoint declares an MTP layer but contains no MTP
tensors, so `auto` must fall back and `required` must fail before readiness.

This is the current product workflow for two-Mac inference. It uses Node Agent
HTTP commands and does not require SSH, machine usernames, passwords, or keys.

## Topology

- Mac A Agent: `http://<node-a-lan-ip>:9100`
- Mac B Agent: `http://<node-b-lan-ip>:9100`
- Mac A: `en4`, `rdma_en4`, `<node-a-rdma-ip>/30`
- Mac B: `en5`, `rdma_en5`, `<node-b-rdma-ip>/30`
- Runtime: `${TOKENITY_RUNTIME_PYTHON}`
- Model: `${TOKENITY_MODEL_ROOT}/Qwen3.5-122B-A10B-4bit`

The retired Mac B address must not be reintroduced.

## After reboot

The installer registers `ai.tokenity.node-agent` with launchd. Confirm both
Agents from the controller Mac:

```bash
curl --noproxy '*' http://<node-a-lan-ip>:9100/v1/node/info
curl --noproxy '*' http://<node-b-lan-ip>:9100/v1/node/info
```

Both responses should report the expected Agent URL, runtime versions, memory,
and active RDMA device. Reinstall Tokenity on a Mac if its Agent is not running;
do not replace the service with an SSH-launched process.

For development, launch the current UI from the stable workspace:

```bash
cd /path/to/Tokenity-Stable
./scripts/run-tokenity-control-app.sh
```

## RDMA preflight

Read the network state through each Agent:

```bash
curl --noproxy '*' -sS http://<node-a-lan-ip>:9100/v1/node/info
curl --noproxy '*' -sS http://<node-b-lan-ip>:9100/v1/node/info
```

Expected values:

- A: `rdma_enabled: true`, `rdma_en4`, `thunderbolt_ip: <node-a-rdma-ip>`
- B: `rdma_enabled: true`, `rdma_en5`, `thunderbolt_ip: <node-b-rdma-ip>`
- no `rdma_errors` on either node

## UI workflow

1. Open **Cluster** and select both Macs.
2. Select **Tokenity Distributed** and **Thunderbolt RDMA**.
3. Select **Create Cluster**.
4. Open **Models** and load `Qwen3.5-122B-A10B-4bit`.
5. Wait for the readiness phase to become `ready` and the row to become
   **Loaded**.

The coordinator calls the worker's typed HTTP rank endpoint. MLX then uses the
Thunderbolt JACCL/RDMA data plane; HTTP does not carry model tensors.

## Direct HTTP start

The UI performs the following request shape:

```bash
curl --noproxy '*' -X POST http://<node-a-lan-ip>:9100/v1/node/start-distributed-openai \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "${TOKENITY_MODEL_ROOT}/Qwen3.5-122B-A10B-4bit",
    "connection_mode": "jaccl",
    "starting_port": 30020,
    "dry_run": false,
    "nodes": [
      {"id":"mac-a","agent_url":"http://<node-a-lan-ip>:9100","lan_ip":"<node-a-lan-ip>","rdma_ip":"<node-a-rdma-ip>","rdma_devices":["rdma_en4"]},
      {"id":"mac-b","agent_url":"http://<node-b-lan-ip>:9100","lan_ip":"<node-b-lan-ip>","rdma_ip":"<node-b-rdma-ip>","rdma_devices":["rdma_en5"]}
    ]
  }'
```

## API verification

After loading completes:

```bash
curl --noproxy '*' http://<node-a-lan-ip>:8000/v1/readiness
curl --noproxy '*' http://<node-a-lan-ip>:8000/v1/models
curl --noproxy '*' -X POST http://<node-a-lan-ip>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-122B-A10B-4bit","messages":[{"role":"user","content":"Say OK."}],"max_tokens":4}'
```

## Stop and cleanup

Stopping the coordinator role also sends an HTTP stop request to worker Agents:

```bash
curl --noproxy '*' -X POST http://<node-a-lan-ip>:9100/v1/node/stop-role \
  -H 'Content-Type: application/json' \
  -d '{"role":"distributed-openai","timeout":10}'
```

Then inspect both HTTP status responses. No distributed role should remain in
the `running` state:

```bash
curl --noproxy '*' http://<node-a-lan-ip>:9100/v1/node/status
curl --noproxy '*' http://<node-b-lan-ip>:9100/v1/node/status
```

See [HTTP Node Agent Protocol](http-node-agent.md) for the control/data-plane
boundary and endpoint security model.
