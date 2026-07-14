# Native MTP real-checkpoint validation — 2026-07-14

This record separates real checkpoint/hardware results from synthetic unit
coverage. All performance values below were measured, not projected.

## Checkpoint provenance

| Component | Pinned revision | `model.safetensors` SHA-256 |
|---|---|---|
| `mlx-community/Qwen3.5-4B-MLX-4bit` target | `32f3e8ecf65426fc3306969496342d504bfa13f3` | `5fb9acd0246866381cf8c5c354c6db1019f6498eec4ccb4f5edcc71ffeacb2db` |
| `mlx-community/Qwen3.5-4B-MTP-4bit` split drafter | `ab6f59bc6627196c611ab8851638651078170485` | `4e0466e99be0114e59a91a834aa4c338c98610e402664a662f97d95fb2da4b28` |

Both upstream repositories declare Apache-2.0. The target download was
3,061,132,920 bytes. The split drafter was approximately 85 MB and its actual
config declares `block_size=4`; Tokenity intentionally requests one draft
only. `scripts/assemble-native-mtp-checkpoint.py` produced:

- target source: <https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit>
- MTP source: <https://huggingface.co/mlx-community/Qwen3.5-4B-MTP-4bit>

- `/Users/Shared/TokenityModels/Qwen3.5-4B-Native-MTP-4bit`
- 29 integrated MTP parameter leaves (31 split source tensors; quantized
  `fc.weight/scales/biases` become one BF16 `fc.weight`)
- `mtp.safetensors` SHA-256
  `76f722e65391fb7f1b021e44ff625798532563a330c35810afec40faea2c2d68`
- static capability `supported`, model type `qwen3_5_text`, declared depth 1

The final target and MTP hashes matched on Mac A and Mac B before inference.

## Hardware and runtime

| Rank | LAN | RDMA device/state | Thunderbolt IP | Python / packages |
|---|---|---|---|---|
| Mac A | `192.168.5.23` | `rdma_en4` active | `192.168.0.1` | shared Python; MLX 0.31.2; mlx-lm 0.31.3 |
| Mac B | `192.168.5.75` | `rdma_en5` active | `192.168.0.2` | shared Python; MLX 0.31.2; mlx-lm 0.31.3 |

The exact interpreter on both ranks was
`/Users/Shared/TokenityRuntime/current/.venv/bin/python`. JACCL tests began
only after both live Node Agents reported the device state and IP shown above.

## World size 1

The real model passed `off`, `auto`, and `required` startup. Four short greedy
requests and five 256-token requests had exact text/reasoning, finish-reason,
and usage parity between standard decoding and Native MTP. Additional real
requests covered stop sequences, maximum tokens, prompt-cache reuse,
temperature/top-p/top-k/min-p sampling, a long generation, client
cancellation, and recovery after cancellation.

For five paired 256-token greedy requests:

| Mode | Completion tok/s | Median tok/s |
|---|---|---|
| standard (`off`) | 111.817, 120.502, 112.686, 119.039, 121.410 | 119.039 |
| Native MTP (`auto`) | 132.115, 134.528, 139.435, 137.943, 142.447 | 137.943 |

Median improvement was 15.88%. The accumulated acceptance snapshot after the
benchmark was 510/761 (67.02%). For one streamed 256-token request, standard
versus MTP was 115.47 versus 125.09 token events/s; TTFT was 321.69 versus
474.60 ms; p50 ITL was 7.519 versus 2.400 ms; p95 ITL was 9.876 versus
11.459 ms.

## World size 2 — Ring

Both ranks published the same decision fingerprint. Rank logs showed the
backbone projection split in half while each rank retained the full MTP
projection shape, providing direct evidence for sharded backbone plus
replicated head.

LAN Ring was communication-bound. A standard 256-token request exceeded the
120-second client deadline and was explicitly stopped through the Agent; both
ranks exited cleanly. Therefore the paired throughput/ITL comparison uses the
same completed 16-token greedy request:

| Metric | standard (`off`) | Native MTP (`auto`) |
|---|---:|---:|
| Output/usage SHA-256 | `6edd923955670ef3403498f02ecb17ef83141f7a45b2e1f1336424e733c4250e` | same |
| Completion tok/s | 1.225 | 1.858 |
| Stream token events/s | 1.290 | 2.148 |
| TTFT | 1604.89 ms | 1571.24 ms |
| p50 ITL | 652.19 ms | 364.35 ms |
| p95 ITL | 756.99 ms | 822.92 ms |
| Observed rank-0 peak RSS | 907.98 MiB | 947.95 MiB |
| Observed rank-1 peak RSS | 965.92 MiB | 1010.78 MiB |

The short completion throughput improved 51.62%, but this is not a general
Ring promotion result because of the short sequence and very slow baseline.
A separate completed 64-token sampling request exercised real mixed cycles.
The cumulative Ring MTP snapshot contained 69 proposals, 64 accepts, and five
rejects (92.75% acceptance), with 1.913 emitted tokens per verify cycle,
5.448 s head time, 53.707 s verify time, and 0.000308 s rollback time. No rank
divergence, cache drift, or deadlock occurred. `required` also started on both
ranks and completed a smoke request.

## World size 2 — JACCL

Three paired 256-token greedy requests produced exact output/finish/usage
hashes in both modes:

- `fa0c18efb1c17fa78b779accdae5da9d0783d831b730a0f9a7315881ef6f2d93`
- `6eeb1f1e47da0d9468bc52f7d78a44e87cb1f40c51cc33d4f3391f32f5758380`
- `dd1c471ad2aaa2f27da843b0258eb44c7f909e43562301b9c769c901c421b38a`

| Metric | standard (`off`) | Native MTP (`auto`) |
|---|---:|---:|
| Completion tok/s, three runs | 123.827, 128.546, 123.733 | 146.656, 140.185, 133.361 |
| Median completion tok/s | 123.827 | 140.185 |
| Stream token events/s | 125.704 | 139.878 |
| TTFT | 125.142 ms | 127.361 ms |
| p50 ITL | 7.077 ms | 8.580 ms |
| p95 ITL | 8.904 ms | 11.348 ms |
| Observed rank-0 peak RSS | 854.95 MiB | 926.94 MiB |
| Observed rank-1 peak RSS | 910.81 MiB | 996.14 MiB |

Median completion throughput improved 13.21%; streamed token-event throughput
improved 11.27%. Per-event p50/p95 ITL regressed because accepted drafts are
emitted in bursts separated by verify cycles, so throughput and individual
inter-event latency must both remain visible.

Across the three non-stream requests plus the matching streamed request, the
MTP snapshot recorded 594 verify cycles, 423 accepted drafts (71.21%), 1.710
emitted tokens per cycle, 0.736 s head time, 5.520 s verify time, and 0.00652 s
rollback time. Real rejection/rollback therefore occurred under JACCL without
deadlock or rank divergence. `required` also started and completed a request.

## Scope limits and cleanup

- Deterministic all-accept/all-reject injection remains synthetic test-only;
  production hardware tests used natural real-model mixed cycles and did not
  add a debug collective or rejection hook.
- Process RSS is the directly observed per-rank process peak, not a claimed
  MLX allocator high-water mark.
- The default remains `off` even though single-rank and JACCL median throughput
  exceeded the 10% future-promotion gate; ITL regressions and broader model
  coverage still require evaluation.
- Every model role was stopped through the HTTP Node Agents. Ports 8000/8010
  had no listeners, no model processes remained, and the original production
  Agent on port 9100 reported an empty role list on both Macs. Temporary test
  Agents on port 9110 were then removed.
