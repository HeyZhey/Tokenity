# Model-load performance validation — 2026-07-14

This validation uses the production TokenityControl path with two 512 GB Macs,
MLX 0.31.2, mlx-lm 0.31.3, and Thunderbolt RDMA/JACCL. The user-provided
comparison point was an approximately 14 GB Qwen3.5 27B checkpoint loading in
about 18 seconds in LM Studio.

## Root cause

The prior Tokenity default materialized one parameter leaf per `mx.eval` and
slept for 50 ms after every leaf. The 2.9 GB model therefore used 924 separate
eval calls and incurred 46.2 seconds of explicit sleep before accounting for
Metal or distributed synchronization. The 65 GB model remained inside the same
loop when it was cancelled after 129 seconds.

## Adaptive policy

The optimized default:

- batches at most 64 parameter leaves;
- bounds each batch to 256 MiB of logical tensor data;
- removes artificial sleep;
- preserves cancellation between bounded batches;
- retains an explicit `fixed` policy for memory-pressure diagnosis.

The runtime defaults to adaptive behavior even when a pre-upgrade Node Agent
forwards the legacy numeric defaults. This lets the on-disk backend optimization
take effect without requiring the root-owned production Agent to restart.

## Measured results

| Model | Inventory size | Before | After | Adaptive batches |
|---|---:|---:|---:|---:|
| Qwen3.5-4B-Native-MTP-4bit | 2.9 GB | 152 s | 8.99 s | 16 |
| Qwen3.5-122B-A10B-4bit | 65 GB | >129 s, cancelled | 20.64 s | 148 |

The 4B load improved by 94.1% (16.9x). The 122B load completed in 20.64
seconds instead of still loading at 129 seconds. Both files had been accessed
previously and macOS reported a large reclaimable file cache, so these figures
primarily validate Tokenity materialization and distributed-start overhead, not
cold-storage throughput.

After loading, `/v1/readiness` reported `ready`, world size 2, and the expected
adaptive batch totals. A UI chat test on the 4B model returned successfully with
0.33-second first response, and an eight-token 122B API smoke request completed
in 0.93 seconds.
