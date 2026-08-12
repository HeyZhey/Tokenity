# MiniMax H3 TP2 半宽 8-bit DQ-GEMM、算子融合与 Tokenity 优化提示词

你现在负责继续优化 Tokenity 中的 MiniMax H3 双 Mac Tensor Parallel 推理。请直接完成代码、测试、双机部署、基准和结果归档；不要只给方案。目标平台是两台相同规格的 M3 Ultra Mac Studio，重点是 TP2 后的半宽 QKV、attention out、FC1、FC2，以及 H3 主干 block 内的 AdaLN/RMSNorm/gate/residual 融合和 MLX materialization 减少。

如果实测证明某条路径不能带来净收益，要保留可复现证据并回退该候选，不要为了“完成优化”默认启用负优化。最终必须给出真实双机数据、正确性证据、改动文件、测试结果和剩余瓶颈。

## 1. 工作目录、分支和保护规则

主要工作目录：

- Tokenity：`/path/to/MinMaxH3/Tokenity-H3`
- 原生 H3/MLX Serve：`/path/to/MinMaxH3/mlx-serve-h3-tp`
- 结果归档根目录：`/path/to/MinMaxH3/outputs`

当前分支：

- Tokenity：`codex/minimax-h3-video`
- 原生端：`codex/minimax-h3-tp2`

两个 worktree 都已有未提交的 Phase 0/1/2 改动。开始前先执行 `git status --short --branch` 并记录基线；这些改动都属于当前工作成果，必须在其上继续，禁止 `git reset --hard`、`git checkout --`、整目录覆盖或清理未跟踪文件。不要修改：

- `/path/to/Tokenity-Stable`
- `/path/to/MLX-Distributed`

原生仓库的 `lib/mlx-src` 和 `lib/mlxc-src` 是当前固定运行时源码。如果性能证据表明必须改 MLX Metal backend，可以在本工作树的固定依赖中做最小修改，但必须：

1. 先证明瓶颈和现有 kernel dispatch；
2. 记录依赖 revision 和完整 diff；
3. 重新构建并随二进制部署匹配的 `libmlx`、`libmlxc` 和 `mlx.metallib`；
4. 不得只替换可执行文件而留下旧 metallib/dylib；
5. 保留非 H3、非 TP2 形状的原始 fallback。

进入原生仓库后先完整阅读 `CLAUDE.md`，涉及测试时同时遵守 `tests/CLAUDE.md`。必须使用 ReleaseFast：

```bash
cd /path/to/MinMaxH3/mlx-serve-h3-tp
./.zig-toolchain/zig build -Doptimize=ReleaseFast
```

`zig build test` 不会替你刷新生产可执行文件；每轮 live A/B 前都要显式重建 ReleaseFast。

不要提交、推送或创建 PR，除非用户另行要求。

## 2. 两台目标机器的实机配置

以下信息已在 2026-08-09 通过 Node Agent 和 SSH 实机核验。不要把序列号、UUID、密码或其他凭据写入源码、日志和结果文件。

| 项目 | Mac A | Mac B |
| --- | --- | --- |
| TP 角色 | Rank 0、coordinator、HTTP/SSE、文本编码、初始 latent、VAE/audio decode、mux | Rank 1、无公网监听、DiT worker |
| 机型 | Mac Studio，`Mac15,14` | Mac Studio，`Mac15,14` |
| 订购型号 | `Z1CE001ALCH/A` | `Z1CE1CH/A` |
| 芯片 | Apple M3 Ultra | Apple M3 Ultra |
| CPU | 32 核：24 Performance + 8 Efficiency | 32 核：24 Performance + 8 Efficiency |
| GPU | 80 核 | 80 核 |
| 统一内存 | 512 GB / 512 GiB | 512 GB / 512 GiB |
| Metal | Metal 4 | Metal 4 |
| macOS | 26.5.1 | 26.5.1 |
| 根卷总容量 | 约 7.999 TB | 约 7.999 TB |
| 核验时可用容量 | 约 7.204 TB | 约 33 GB，属于硬约束，运行前重新查询 |
| SSH | `<node-a-user>@<node-a-lan-ip>` | `<node-b-user>@<node-b-lan-ip>` |
| 本次 H3 Node Agent | `http://<node-a-lan-ip>:9100` | `http://<node-b-lan-ip>:9200` |
| Thunderbolt/RDMA IP | `<node-a-rdma-ip>` | `<node-b-rdma-ip>` |
| 活跃 RDMA 设备 | `rdma_en4` | `rdma_en5` |
| TP 数据面 | JACCL strict ring | JACCL strict ring |

两台 H3 Agent 核验时共同使用：

- 架构：`arm64`
- MLX：`0.32.0`
- mlx-lm：`0.31.3`
- Tokenity：`0.1.0`
- H3 Tokenity code revision：`e2fd7a94e8e626c739427c48f62caaae6c636384660687f8f281fc1cd8161d6d`

重要拓扑说明：

1. 历史 Tokenity 文档、fixture 和 Swift 默认值仍大量记录 Mac A `<node-a-lan-ip>:9100`，但该地址在本轮核验时 Node Agent 与 SSH 都超时；成功的 H3 TP2 硬件基线使用的是 `<node-a-lan-ip>:9100`。任何部署前都必须重新访问 `/v1/node/info`，结合 SSH hostname、芯片、内存、RDMA IP/设备证明节点身份，禁止仅凭旧 IP 猜测。
2. Mac B 的 `:9100` 当前是另一套长期 Tokenity Agent，code revision 为 `453a402ab8268d20f9f7201b6665e68664e2ee5ea0b514978792b8ad5312168d`，不声明 `minimax_h3_video` capability；本次 H3 分支使用 `:9200`。不要覆盖、重启或误用 `:9100` 上的稳定服务。
3. Mac B 根卷核验时只剩约 33 GB。不要在 B 创建完整模型副本、重复 shard、巨型 trace 或无限日志；构建、长 trace 和统一结果优先放在 Mac A，再只向 B 部署必要的 Rank 1 shard 和匹配运行时。部署前做容量 gate，完成后清理本任务明确创建且已归档的临时产物。
4. Mac B 上还存在一个早于 Node Agent 的单机 H3 进程，核验时约 320 MiB RSS、0% CPU。它不是本任务创建的进程，不能无范围地 kill；正式基准前要证明它没有占用 GPU/生成请求，若确需停止要按精确 PID/命令记录处理，并在报告中说明。
5. M3 Ultra 目标走普通 Metal/Steel 路径，不要把 M5 NAX/Neural Accelerator 的结果混入本次结论。运行时明确记录 `is_nax_available()`；本任务的优化必须在两台 M3 Ultra 上成立。

## 3. 当前模型、checkpoint 和运行时合同

本地模型源：

`/path/to/Codex/2026-08-03/minimax-h3-https-hf-mirror-com/models/MiniMax-H3-FL2VA-MLX-Serve-8bit`

已部署模型根目录：

`${TOKENITY_H3_MODEL_PATH}`

已部署 Phase 2 原生端：

`${TOKENITY_H3_BINARY_PATH}`

模型和 TP2 合同：

- MiniMax H3 affine 8-bit，`group_size=64`，计算 dtype 为 BF16；不是 4-bit，也不是 BF16 权重版。
- hidden size：`5376`
- 主干 block：`50`
- attention：`56` heads，head dim `128`，完整 inner dim `7168`
- TP2：每 rank `28` heads，local attention inner dim `3584`
- FFN：完整 `14336`，每 rank `7168`
- 每个 block 恰好两个 all-sum：attention out projection 后一次、FC2 后一次。
- 标准 28-step、`fast=false` 请求：`50 × 2 × 28 = 2800` 次有序 all-sum。
- Rank 0 拥有 API、文本编码和媒体后处理；Rank 1 只参与主 DiT。
- 分布式协议版本固定为 `1`，控制帧包含 magic/version/epoch/sequence/command/integrity。不得在模型 collectives 中插入无序的异步控制 collective。
- checkpoint manifest schema/protocol/world size、rank shard size 和 SHA-256 必须在 readiness 前严格一致。

当前已验证 artifact：

- 模型 revision：`b213ba3882a8262d67e735f9f93cb962e051a2d6e4579fdd07e928f0a7b77dab`
- Rank 0 transformer shard SHA-256：`f4757cef76f6fade60dc5532a12ccf55714e51d760bf1b2299995ba2784ed45c`
- Rank 1 transformer shard SHA-256：`e252b80a7810bc8e52d2066c4e68f5f7830066d1e53b3b3d055249e480a70754`
- 当前最终基线 binary SHA-256：`b4459a15d00f0319347bcbf8f309b642d8fcaaf91e2e8a6b5956f4b013929a5c`

每个新候选必须同时记录可执行文件、`libmlx`、`libmlxc`、`mlx.metallib`、Tokenity code 和 checkpoint shard 的 fingerprint。两 rank 任一不一致时 fail closed，不进入性能测试。

## 4. 固定基准与当前性能

所有正式比较使用完全相同的请求：

```json
{
  "prompt": "A cinematic tracking shot of a paper boat crossing a rainy neon street",
  "width": 512,
  "height": 256,
  "num_frames": 124,
  "steps": 28,
  "seed": 42,
  "fast": false,
  "stream": true
}
```

注意用户也会把分辨率口头写成 `256×512`，API 中固定为 `width=512,height=256`。此 prompt 的实际 packed sequence 是 `5163` 行；独立 collective gate 使用规范形状 BF16 `[5150,5376]`。微基准必须同时覆盖 `M=5150` 和真实 `M=5163`，不能只为单个 prompt 长度写死一个精确 M。

2026-08-09 相同实例三次复测：

| 运行 | 温度/缓存 | 端到端 | Rank 0 DiT | Rank 1 DiT | Rank 1 每步均值 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | cold file cache | 204.383 s | 187.173 s | 187.146 s | 6680.786 ms |
| 2 | warm | 200.664 s | 187.257 s | 187.246 s | 6684.607 ms |
| 3 | warm | 200.435 s | 187.235 s | 187.234 s | 6683.964 ms |

聚合基线：

- warm 端到端均值：`200.549842 s`
- warm DiT：约 `187.246 s`
- 全部 step 均值：`6683.119 ms`，总体标准差 `8.483 ms`
- warm 两次端到端 range：`0.228764 s`
- 单机参考：DiT `301.891 s`、端到端 `322.583 s`
- TP2 相对单机：DiT `1.612×`、warm 端到端 `1.608×`

独立 2800 次 collective gate 的较慢 rank：

- total：`19.498716 s`
- p50：`6.943 ms`
- p95：`6.991 ms`
- effective bandwidth：`7.405 GiB/s`
- correctness：两个 rank 都为 `3.0`
- 两个 rank 都 clean shutdown

当前瓶颈估算：

- communication：约 `19.499 s`，占 DiT `10.41%`
- local DiT compute：约 `167.723 s`，占 DiT `89.59%`
- warm 非 sampling：约 `13.304 s`，其中 video decode 约 `9.566 s`

本任务首要目标是本地 DiT compute 至少减少 `10%`，即约节省 `16.8 s`。在 collective 不回退的情况下，目标应接近：

- DiT：`≤ 170.5 s`
- warm 端到端均值：`≤ 184.0 s`（推算约 `183.8 s`）

最终的收益必须来自实测，不得直接用独立 collective 数字从端到端中相减后宣称完成。应增加足够细的原生 operator/kernel 计时，区分本地计算、collective wait、强制 materialization 和媒体后处理。

当前 TP2 三次输出彼此完全一致，golden：

- RGB8 video，124 帧，`48,758,784` bytes，SHA-256 `e5c486ed9071c11c6393027f5f5cca19e70e20bbbc69f07f224a871fadc759f0`
- stereo PCM audio，`662,400` bytes，SHA-256 `979488ec238bffb3b1e4b9bb241875848508b21c465982d51e773afc7d2d3850`

原始基准证据：

- `/path/to/MinMaxH3/outputs/tp2-run13-three-repeat-analysis.json`
- `/path/to/MinMaxH3/outputs/tp2-run9-512x256-124f-28steps.json`
- `/path/to/MinMaxH3/mlx-serve-h3-tp/docs/minimax-h3-tp2-collective-benchmark.md`
- `/path/to/MinMaxH3/Tokenity-H3/docs/minimax-h3-video.md`

## 5. 必须先理解的当前实现

不要重新实现 Phase 2。先沿现有路径定位：

- `src/minimax_h3.zig`
  - `attnForward`：QKV、Q/K RMSNorm、RoPE、transpose/contig、SDPA、out projection、TP all-sum
  - `mlpForward`：FC1、split gate/up、SiLU、multiply、FC2、TP all-sum
  - `modScaleShift`：按 `TimestepPlan.runs` 做 RMSNorm 后的 AdaLN scale/shift
  - `modGate`：按 run 做 `residual + gate * branch`
  - `Model.forward`：50 个主 block 的完整调用和 array 生命周期
- `src/mage_flow.zig`
  - `MfLinear.forward`
  - `MfLinear.dqGemmWide`
- `src/h3_distributed.zig`
  - `allSum` 的跨 stream materialization 和 JACCL 边界
- `scripts/shard-minimax-h3-tp2.py`
- `lib/mlx-src/mlx/backend/metal/quantized.cpp`
  - `qmm`、`qmm_splitk`、`QuantizedMatmul::eval_gpu`
- `lib/mlx-src/mlx/backend/metal/matmul.cpp`
  - Steel GEMM tile、split-K、swizzle dispatch
- `lib/mlx-src/mlx/backend/metal/kernels/quantized.h` 及关联 Metal kernel
- `lib/mlxc-src/examples/example-metal-kernel.c` 和当前 Zig FFI，用于评估 `mlx_fast_metal_kernel` 实现方式

当前名称为 `dq-gemm` 的 `MfLinear.dqGemmWide` 实际流程是：

1. 对完整 packed affine weight 调 `mlx_dequantize`；
2. 对完整解量化权重 transpose；
3. 使用普通 `mlx_matmul`/Steel GEMM；
4. 每次 forward 都构建该 lazy 路径，完整 DQ tensor 随调用释放。

它不是“在同一个 GEMM kernel 内按 tile 解量化”的真融合，只是 DQ + Steel GEMM 路线。它在 H3 480p 曾相对 stock qmm 带来约 `13%/step` 的收益，所以目前 H3 默认 floor 为 2048，并有 `[mf-linear] dq-gemm engaged` 日志。

反过来，MLX stock `qmm_t` 本身会边读量化权重边计算，但 M3 路径当前大 M qmm 使用固定的 `bm=32,bn=32,wm=2,wn=2`，并由通用 split-K heuristic 决策；它没有针对下面四个 TP2 半宽形状校准。不要混淆这两个路径，也不要仅凭函数名宣称已经完成 dequant+GEMM 融合。

## 6. 真实 TP2 半宽 GEMM 形状矩阵

对于实际 `M≈5150–5163`、BF16 activation、affine 8-bit/group 64，四个主线性层每个 rank 的逻辑 GEMM 是：

| 名称 | activation | logical weight | output | packed u32 weight | scales/biases |
| --- | --- | --- | --- | --- | --- |
| QKV | `[M,5376]` | `[5376,10752]` | `[M,10752]`，即 Q/K/V 各 3584 | `[10752,1344]` | `[10752,84]` |
| Attention out | `[M,3584]` | `[3584,5376]` | `[M,5376]` | `[5376,896]` | `[5376,56]` |
| FC1 gate+up | `[M,5376]` | `[5376,14336]` | `[M,14336]`，即 gate/up 各 7168 | `[14336,1344]` | `[14336,84]` |
| FC2 | `[M,7168]` | `[7168,5376]` | `[M,5376]` | `[5376,1792]` | `[5376,112]` |

这里的 `biases` 是 affine quantization 的每组 offset，不是输出层的普通 add bias，不能在 epilogue 中错误处理。QKV 和 FC1 shard 必须在每个 Q/K/V 或 gate/up 语义分段内切 rank；out projection 和 FC2 在输入列维度切分。不能把 raw packed rows/columns 当成普通 dense tensor随意对半。

每个 rank、每个 denoise step、每个主 block 都会跑这四个 GEMM；标准请求相当于每种形状每 rank 各 `28×50=1400` 次。微小的单次收益会被大量放大，因此必须按单 kernel 和完整 block 两个层次测量。

## 7. 实施顺序

### 7.1 冻结、复现和增加可观测性

先在未改优化代码前完成以下基线：

1. 两个 Node Agent 的 `/health`、`/v1/node/info`、身份、内存、磁盘、RDMA active、Thunderbolt IP、版本与 capability。
2. 两 rank 的 binary/runtime/metallib/checkpoint fingerprint 一致性。
3. 独立 2800-collective gate。
4. 固定请求至少一次 cold + 两次 warm，确认落在现有噪声范围并复核 golden hashes。
5. 记录两 rank 温度、功耗/频率可获得指标、后台 GPU 占用和 memory peak，避免把热降频或后台任务当成 kernel 回退。

增加结构化、默认低开销的 H3 性能统计，至少能区分：

- QKV、attention out、FC1、FC2；
- weight DQ、transpose/contiguous copy、GEMM；
- RMSNorm + AdaLN modulation；
- Q/K norm/RoPE/transpose；
- SDPA；
- SiLU/gate-up；
- residual gate；
- `mlx_array_eval(partial)`、JACCL all-sum、`mlx_array_eval(output)`；
- block、step、rank 总时长和 peak memory。

计时必须显式 materialize 被测输出，避免只测 lazy graph 构建。性能模式可通过仅用于工程 A/B 的 kill switch 开启，默认生产路径不能因细粒度同步而变慢。每个优化 arm 必须在自己的日志/JSON 中输出明确的 engagement count，不能从环境变量或输出相同推断它实际命中。

### 7.2 建立四形状微基准

为原生端增加可重复的 H3 TP2 kernel microbenchmark，建议是一个 typed CLI/subcommand 和 JSON 输出，不要依赖 Python 去计时 MLX lazy 调用。它至少比较：

1. stock `mlx_quantized_matmul`/qmm；
2. 当前 `dequantize + transpose + Steel GEMM`；
3. 一次性预解量化并缓存 dense-transposed weight（诊断 arm，必须报告额外常驻内存）；
4. 新的 fused tiled DQ-GEMM 候选。

每个形状分别预热、运行足够次数、显式 eval，报告 p50/p95/mean、有效吞吐、临时和常驻内存、kernel 名称、tile 参数、split-K、输出误差与 engagement。覆盖 `M=5150`、`5163`，并至少用附近不同 prompt 长度验证 dispatch 不会在 M 轻微变化时失效。

不要从端到端一个数字直接选择 tile。对 M3 Ultra 80-core GPU 搜索并记录合理的 `BM/BN/BK`、`WM/WN`、threadgroup、split-K、swizzle 候选；避免只为了这四个 N/K 写散落的 magic number。最终选择应集中为一个可审计的 M3/H3/TP2 shape profile，并在未知硬件、dtype、bits、group size、对齐或形状上回退。

### 7.3 真正融合的 8-bit DQ-GEMM

目标 kernel 在 K tile 加载 packed u32 weight、scale 和 affine offset，在寄存器/threadgroup tile 内解包并立刻进入 MMA/GEMM，不生成完整 `[K,N]` BF16 DQ tensor，也不生成完整转置副本。要求：

- 首先支持本任务真正使用的 affine 8-bit、group 64、BF16 activation/output、transpose-weight 语义；
- 正确处理 K/N 边界和对齐，不能只在一个 M 上工作；
- 对 QKV/FC1 的大 N 和 out/FC2 的不同 K/N 分别选择实测最优 profile；
- 允许融合普通输出 bias epilogue，但不能把 quantization offset 当作 output bias；
- 不支持的 bits、group、dtype、shape、硬件全部走已有路径；
- kernel/pipeline 要缓存，不能在 1400 次调用中重复编译；
- 明确记录命中的 layer/shape、tile 和次数；
- 如果直接改 MLX qmm 优于新增外部 custom kernel，允许选择 MLX backend 方案，但必须保证非 H3 通用形状不回退并有测试；
- 如果一次性预解量化 cache 反而更快，可以作为独立候选保留，但必须准确更新 Tokenity 的每 rank memory reservation，证明 512 GB 内存峰值和重复请求无增长。不能把额外约数十 GB 常驻内存隐瞒在“kernel 优化”中。

最终默认路径应由数据决定。理想结果是 fused DQ-GEMM 同时击败 stock qmm 和当前全量 DQ + Steel，且显著降低临时内存/带宽；若做不到，保留当前路径并提交 no-go 数据，不默认启用负优化。

### 7.4 H3 block 算子融合

按 profile 占比逐项实施，至少评估以下三个融合点：

1. `RMSNorm + AdaLN scale/shift`
   - 当前 `rmsNormLast` 后，`modScaleShift` 对每个 run 做 slice、`1+scale`、multiply、add，最后 concat。
   - 设计一个按 row 找到 `mod_row` 的融合路径，在同一 kernel 内完成 hidden=5376 的 RMS reduction、norm weight、scale 和 shift，避免为每个 run 建图/切片/拼接。
   - 必须严格遵守 `TimestepPlan.runs`、三种 modality tag 和 norm epsilon。

2. `gate + residual`
   - 将 `x + other * gate[mod_row]` 合为一个 elementwise kernel，直接写完整 `[M,5376]`，消除每 run slice/mul/add/concat。
   - attention cache/broadcast 非 refresh step 只重算当前 gate 的 PAB 语义必须保持不变。

3. `FC1 split + SiLU + gate/up multiply`
   - 将 `[M,14336]` 的语义 split 与 `silu(gate)*up` 合为一个输出 `[M,7168]` 的 kernel，避免两个 view/array 和中间激活 materialization。
   - gate/up 在 shard 中的语义顺序必须与 sharder一致。

可在证据充分后进一步评估 Q/K RMSNorm + partial RoPE + transpose/contiguous fusion，但不要让它阻塞上面三项。每个融合都必须有独立 kill switch、engagement counter、microbenchmark 和 block-level A/B；收益太小或使数值/维护风险显著上升的候选不默认启用。

### 7.5 减少 block 间 MLX materialization

系统性审计 `Model.forward`、`attnForward`、`mlpForward` 中的：

- `contig`、transpose 后再 contig；
- split/slice/concat 造成的复制；
- dtype cast；
- array retain/free 导致的意外 graph lifetime；
- 显式 `mlx_array_eval`；
- all-sum 前后的 buffer ownership 和 stream 交接。

目标是在两个 collective 边界之间维持可融合的 lazy graph，减少中间完整 `[M,N]` 数组和重复 contiguous copy。需要用 trace/counter 证明 dispatch 数、copy bytes、临时峰值确实下降。

`h3_distributed.allSum` 目前在 CPU communication stream 上工作，先 `mlx_array_eval(partial)`，再 JACCL all-sum，随后 `mlx_array_eval(output)`；这是为避免跨 stream buffer 注册竞态而加入的正确性边界。不要盲目删除。若要改变它，必须先建立双 rank 压力测试，证明连续多 block、多请求、取消和 shutdown 下没有 allocation race、collective 次序错误或死锁。否则保留两个强制边界，把优化集中在边界之间。

## 8. Tokenity 侧必须完成的优化与集成

kernel 逻辑属于原生端，不能搬到 Tokenity Python。Tokenity 负责安全、可重复地部署和验证该优化：

1. 保持已有 `start-minimax-h3-video` typed orchestration、Rank 0/1 角色、2/2 quorum、protocol v1、SSE 透传、request lease、rollback 和 coordinated stop。
2. 增加一个可复用的双机 H3 benchmark/deploy harness，自动完成 dry-run/preflight、指纹校验、启动、readiness、固定请求、原始 SSE/日志抓取、两 rank timing 合并、hash、stop、资源账本检查和 JSON 归档。不要继续依赖一次性的手工 curl 和记事。
3. 节点拓扑必须来自 request/config 并用 `/v1/node/info` 实时验证，不要把 `.14`、`.23`、`:9200` 继续扩散为新的业务层 IP heuristic。旧 `.23` 和当前 `.14` 的迁移要在文档/默认配置中明确收敛，但不得破坏测试 fixture 表达的通用双节点行为。
4. readiness evidence 增加或严格核验 binary、MLX dylib、metallib、checkpoint/model revision、kernel profile 和 optimization flags 的一致性；rank 不一致时 HTTP `412` fail closed。
5. 只允许 typed、allowlisted 的 H3 性能 profile/kill switch。禁止通过 HTTP 接收任意环境变量、任意 shell 或任意路径写入。
6. 结构化保存两 rank 的 kernel engagement、phase timing、collective timing、peak memory、磁盘余量和 thermal/background note；不能只解析人类日志中的一个总秒数。
7. 如果启用预解量化 cache 或其他增内存方案，更新 Rank 0/1 的 `memory_reservation_breakdown`，准入检查要使用新峰值，stop 后 reservation、port、process 必须归零。
8. Mac B 低磁盘必须成为部署前 gate：预估部署/trace 空间不足就拒绝并给出清理建议，不允许写到磁盘满后再失败。
9. 保持 Rank 1 无公共 H3 监听、Rank 0 才拥有外部 API；不要干扰 Mac B `:9100` 的稳定 Tokenity Agent。
10. 出现任一 rank 启动/校验/benchmark 失败时回滚本次实例，不得留下单 rank 卡在 collective、孤儿进程、端口或 memory reservation。

## 9. 测试和正确性门槛

遵守原生仓库的 TDD 规则：每个行为改动先有能在旧实现上因正确原因失败的测试，再做最小实现，最后跑完整相关套件。

### 9.1 Kernel/单元测试

至少覆盖：

- 四个真实 TP2 形状的 8-bit/group-64 dispatch；
- `M=5150`、`5163` 和附近非精确 M；
- K/N 对齐与尾部处理；
- packed weight、scale、affine offset 语义；
- Q/K/V 与 gate/up semantic shard 顺序；
- fused DQ-GEMM vs 当前 DQ+matmul、stock qmm 和可承受尺寸的 FP32 ground truth；
- RMSNorm+AdaLN 的所有 modality/run 映射；
- gate+residual；
- fused SwiGLU；
- 不支持 shape/dtype/bits/group/hardware 时的 fallback；
- engagement counter 确实只在命中时增长；
- NaN/Inf、空/非法 shape 拒绝；
- kernel cache 重用与清理。

数值门槛不能只看 cosine。至少记录 max absolute、max relative、RMSE、cosine，并要求新 fused 路径相对 FP32 reference 的误差不劣于当前生产路径；任何放宽都要有定量原因。优先保持 BF16 输出一致。

### 9.2 原生和 Tokenity 回归

至少运行：

- `./.zig-toolchain/zig build test -Doptimize=ReleaseFast` 或仓库要求的完整 `zig build test` 组合；
- MiniMax H3 相关 filter/live fixture；
- `tests/test_minimax_h3.sh` 中适用的 e2e/SSE/400/snapping 项；
- `pytest` 覆盖 `tests/test_shard_minimax_h3_tp2.py`；
- Tokenity 的 `tests/test_minimax_h3_video.py`、相关 `tests/test_node_agent.py`、CLI/supervisor 回归；
- 任何新增 benchmark/deploy harness 的 hermetic 测试；
- 原生 `--help`/capability/protocol 参数回归。

### 9.3 双机正确性和生命周期

每个准备进入最终比较的候选必须：

1. 通过 2800-call collective gate，较慢 rank total 不得比 `19.499 s` 回退超过 5%，p95 不得显著恶化；
2. 达到 2/2 readiness，两个 rank 的所有 fingerprint/profile 一致；
3. 固定请求 HTTP 200，31 个预期 progress/complete 事件，返回 124 帧和完整 stereo PCM；
4. 同候选连续三次相同 seed 的 video/audio hash 彼此一致；
5. 优先要求与当前 golden video/audio SHA-256 字节一致。若新 GEMM reduction order 造成非字节一致，该候选不得悄悄成为默认：先保存 raw artifact，报告逐帧 max/mean error、PSNR/SSIM、音频 max error/SNR 和 latent/step drift，并取得用户明确同意后才可放宽发布门槛；
6. 覆盖重复请求、客户端中断后的下一请求、rank failure rollback 和 coordinated stop；
7. 两 rank exit code 都为 0，无 last_error、孤儿、端口、lease 或 memory reservation 泄漏；
8. 多轮后内存不单调增长，Mac B 磁盘不被 trace/log 持续吃满。

## 10. 性能实验方法与验收

微基准必须先筛掉负优化，完整双机只测有希望的候选。最终基线与候选使用同一构建/模型/拓扑/客户端，进行受控配对 A/B：

- 至少一个 cold 运行用于启动/文件缓存信息，不混入 warm 结论；
- 最终每个 arm 至少 3 个 warm 样本，交错或平衡运行顺序，避免热状态和 boot order 偏差；
- 同一 arm 的日志必须证明开关和 kernel engagement；
- 报告两 rank，不只报告快的 rank；
- 保存每 step 分布、mean/p50/p95/min/max 和 rank skew；
- 同时报告端到端、DiT、本地四 GEMM、融合算子、collective、decode 和内存；
- 如 thermal 状态失控，暂停冷却后重做，不能挑最快单次。

发布目标：

- 主要目标：estimated/measured local DiT compute 至少提升 `10%`；
- 端到端目标：warm mean `≤184.0 s`；
- DiT 目标：约 `≤170.5 s`；
- collective 不回退超过 5%；
- 结果确定性、质量、协议和生命周期全部通过；
- 不以异常增大的常驻内存、Mac B 磁盘耗尽或关闭正确性检查换速度。

分项候选若收益小于约 3% 且显著增加维护复杂度，不应默认启用；但多个简单、可验证的融合可按累计收益评估。即使未达到 10%，也必须把已验证的正收益、no-go 路径和新的瓶颈占比完整报告，不能伪造达到目标。

## 11. 结果归档格式

为每个最终 arm 在 `/path/to/MinMaxH3/outputs` 创建独立、可追溯的 JSON 和原始日志引用，至少包含：

- test id、日期、prompt/request；
- Mac A/B 实时硬件、OS、内存、磁盘、Agent、LAN/RDMA 拓扑；
- git status/revision 和所有 runtime/checkpoint fingerprints；
- optimization profile、kill switches、tile 参数和 engagement count；
- microbenchmark 每形状结果；
- cold/warm 双 rank phase/step/operator 数据；
- collective gate；
- video/audio 字节数、hash 和必要的质量指标；
- memory/temporary/copy/materialization 指标；
- readiness、取消、重复请求、rollback、shutdown、resource ledger；
- 与 `tp2-run13-three-repeat-analysis.json` 的绝对差、百分比和置信说明。

同时更新原生 H3 TP2 文档和 Tokenity H3 文档，使当前拓扑、kernel profile、使用方法、fallback、已验证数据和限制一致。不要删除历史基线；标注日期和方法。

## 12. 最终交付内容

完成后用中文交付：

1. 一句话结论：是否达到本地 compute +10% 和 warm e2e ≤184 s；
2. Mac A/B 最终实时配置和拓扑；
3. 根因和最终选择的 kernel/fusion 路径；
4. 四个半宽 GEMM 的 before/after 表；
5. 三次以上 warm 双机端到端、DiT、collective、rank skew 表；
6. 正确性 hash/质量、确定性和生命周期证据；
7. 内存、临时 buffer、materialization/copy、Mac B 磁盘变化；
8. Tokenity 的部署、准入、指纹、观测和 rollback 改动；
9. 修改文件清单与测试命令/结果；
10. 被否决的候选及证据；
11. 当前剩余最大瓶颈和下一步预估上限；
12. 所有 JSON、日志、视频和文档的绝对路径。

不要只贴计划、伪代码或单机 microbenchmark。除非遇到确实需要新权限的阻塞，否则持续执行到双 M3 Ultra 上有完整、可复现、可回退的最终结果。
