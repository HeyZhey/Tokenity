# Tokenity Stable 全面审计、稳定性、性能与多模型实例改造提示词

你是 Codex。请作为资深 Apple Silicon / MLX / MLX-LM 推理工程师、分布式系统架构师、macOS SwiftUI 工程师和性能工程师，在 Tokenity 稳定仓库中完成一次以证据为基础的全面审计、Bug 修复、性能优化和架构升级。

这不是只写分析报告，也不是只修表面状态。你需要持续推进到代码、测试、基准和可复现验证均完成。遇到问题先定位根因，不得用延时、假状态、吞异常、自动重试死循环或 UI 文案掩盖问题。

## 唯一正确的项目目录

主项目和唯一事实来源：

```text
/Users/zxc/Documents/Tokenity-Stable
```

所有读取、修改、测试、构建、打包和启动均以此仓库为准。开始前先执行 `pwd` 并确认路径；不得在名称相似的旧工作区中工作。

正确的开发 UI 启动方式：

```bash
cd /Users/zxc/Documents/Tokenity-Stable
./scripts/run-tokenity-control-app.sh
```

严禁从以下旧路径构建或启动 UI：

```text
/Users/zxc/Documents/MLX-Distributed/apps/TokenityControl
```

旧 `MLX-Distributed` 目录只允许作为只读参考资料来源。不得修改、构建或启动其中的 Tokenity UI/后端，也不得把旧二进制误认为当前稳定仓库的产物。

只读参考项目：

- oMLX：`/Users/zxc/Documents/MLX-Distributed/omlx-main-0713`
- exo：`/Users/zxc/Documents/MLX-Distributed/exo-main-0713`

参考项目只能用于理解成熟实现、状态机、生命周期、调度、监控和测试思路。不要整体移植，不要复制品牌资产或无关 UI。若确实复用受许可证保护的代码，先核对许可证并保留所需版权、NOTICE 和修改说明；优先复用设计思想并在 Tokenity 中适配实现。

## 保护当前工作树

当前稳定仓库可能已有用户未提交修改和新文件，尤其涉及 Chat、MenuBar、TokenityStore、Node Agent、distributed runtime 及其测试。这些都属于用户现有工作：

- 开工前记录 `git status --short` 和相关 `git diff`。
- 不执行 `git reset --hard`、`git checkout --`、`git clean` 或任何会丢失用户改动的操作。
- 不覆盖、删除或回退用户已有的 Chat/MenuBar/状态修复。
- 修改与现有改动重叠的文件时，先理解当前 diff，再做最小兼容编辑。
- 每个阶段运行 `./scripts/verify-stable-baseline.sh` 或等价的针对性测试加全量回归。

进入任何目录前读取并遵守适用的 `AGENTS.md`、项目规则和构建说明。

## 固定环境与已验证拓扑

### Mac A

- SSH 调试地址：`apple@192.168.5.23`
- Node Agent：`http://192.168.5.23:9100`
- 普通网络接口：`en4`
- RDMA 设备：`rdma_en4`
- Thunderbolt/RDMA IP：`192.168.0.1`

### Mac B

- SSH 调试地址：`probriefing@192.168.5.75`
- Node Agent：`http://192.168.5.75:9100`
- 普通网络接口：`en5`
- RDMA 设备：`rdma_en5`
- Thunderbolt/RDMA IP：`192.168.0.2`

Mac B 的唯一当前普通网络地址是 `192.168.5.75`。任何旧 Mac B LAN 地址都属于废弃配置：

- 不得恢复旧地址。
- 不得在默认节点、fallback、迁移代码、sample、测试 fixture、文档、脚本或 UI 中重新加入旧地址。
- 不得因为旧缓存或旧构建产物将 Mac B 回写到其他地址。
- 对 Mac B 身份的兼容逻辑只能收敛到 `192.168.5.75`；不要保留陈旧地址 alias。
- `192.168.0.2` 是 Mac B 的 Thunderbolt/RDMA 数据面地址，不是废弃 LAN 地址。

### 模型与 Runtime

模型路径：

```text
/Users/Shared/TokenityModels/Qwen3.5-122B-A10B-4bit
```

Runtime Python：

```text
/Users/Shared/TokenityRuntime/current/.venv/bin/python
```

期望运行版本：

- `mlx == 0.31.2`
- `mlx-lm >= 0.31.3`

分布式启动前必须通过每个 Node Agent 或 Codex 诊断命令验证：Python 路径、MLX/MLX-LM 版本、模型目录、模型 revision/关键文件、RDMA device/IP 和 Node Agent 代码版本。所有 rank 的 MLX 版本必须一致；MLX-LM 版本必须兼容，不能仅用字符串比较。版本不符合时显示明确的 readiness 阻断原因，不得继续启动后表现成卡死。

## SSH 的严格边界

SSH 只允许作为 Codex/开发者的带外调试手段，不属于 Tokenity 产品架构。

### Codex 允许使用 SSH 的场景

- 检查机器是否可达、确认 hostname。
- 读取诊断信息：版本、进程、端口、Agent 日志、系统/RDMA 状态、文件 hash。
- 在 Node Agent 未运行且需要继续调试时，使用本提示词给出的精确命令临时拉起 Agent。
- 执行明确的故障注入或清理命令，但必须说明影响，并避免破坏用户数据。

基础 SSH 可达性检查：

```bash
ssh apple@192.168.5.23 hostname
ssh probriefing@192.168.5.75 hostname
```

如果确认 Mac A 的 Node Agent 未运行，可在调试期间临时启动：

```bash
ssh apple@192.168.5.23 'mkdir -p "$HOME/Library/Application Support/Tokenity/logs"; nohup env PATH=/Users/Shared/TokenityRuntime/current/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin PYTHONPATH=/Users/Shared/TokenityCode /Users/Shared/TokenityRuntime/current/.venv/bin/python -u -m tokenity node-agent --host 0.0.0.0 --port 9100 > "$HOME/Library/Application Support/Tokenity/logs/node-agent.log" 2>&1 &'
```

如果确认 Mac B 的 Node Agent 未运行，可在调试期间临时启动：

```bash
ssh probriefing@192.168.5.75 'mkdir -p "$HOME/Library/Application Support/Tokenity/logs"; nohup env PATH=/Users/Shared/TokenityRuntime/current/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin PYTHONPATH=/Users/Shared/TokenityCode /Users/Shared/TokenityRuntime/current/.venv/bin/python -u -m tokenity node-agent --host 0.0.0.0 --port 9100 > "$HOME/Library/Application Support/Tokenity/logs/node-agent.log" 2>&1 &'
```

运行上述命令前先检查 9100 端口和现有 Agent，防止重复启动。临时 SSH 拉起只用于恢复调试，不得作为产品成功验收依据；正式安装/重启路径仍应由 installer/launchd 管理。

### 产品代码中绝对禁止的 SSH 用法

- SwiftUI App 不得执行 SSH。
- Node Agent 不得通过 SSH 启动另一台 Mac 的 rank、runtime 或模型。
- coordinator 不得 self-SSH，也不得 SSH 到 worker。
- launcher、hostfile、supervisor、模型加载和动态实例调度不得依赖 SSH 用户名、密钥、密码或远程 shell。
- 不得把 `ssh` 字段、`user@host` 或 SSH 命令重新引入生产启动 DTO。
- 不得通过 SSH 复制部署代码、同步模型或把调试命令固化进项目工作流。

Tokenity 产品的边界必须保持：

- 控制面：SwiftUI/Coordinator 通过 Node Agent typed HTTP API（端口 9100）编排。
- 每台 Agent 只监督本机 rank/实例进程。
- 数据面：MLX distributed Ring 或 JACCL/RDMA；HTTP 不传模型 tensor。
- 远程 rank 启动：coordinator 调用 worker Node Agent 的 typed HTTP rank endpoint。
- SSH 只存在于 Codex/开发者调试终端，不能存在于产品调用链。

审计所有 launcher、legacy hostfile、环境变量和 fallback。任何生产分支出现 SSH 启动都必须移除或永久禁用，并加回归测试；不要为了兼容旧实现恢复 SSH 组网。

## 总目标

在不破坏任何现有可用功能、不重新设计 UI 的前提下，完成以下目标：

1. 单机模型推理稳定可用。
2. 多机 Standard Network/Ring 与 JACCL/RDMA 推理稳定可用。
3. 模型状态、菜单栏状态、节点状态和内存状态全部来自同一套真实运行状态，不再出现假 loaded 或绿灯与实际不符。
4. 支持多个相互独立的模型实例动态加载、运行、路由和卸载。
5. 支持混合部署：例如 Mac A + Mac B 通过 RDMA 运行一个分布式 GLM 模型，同时 Mac A 还可以运行一个单机 Qwen 模型；前提是资源准入检查通过。
6. 优化冷启动、TTFT、prefill、decode、并发吞吐和长上下文体验，并用基准数据证明收益。
7. 提升崩溃恢复、取消、超时、清理、并发安全和错误可观测性。

若缺少真实硬件，只能将对应项标记为“等待硬件验证”，同时完成自动化测试、故障注入脚本和逐步人工验证清单；不得声称已在真实硬件上通过。

## 不可妥协的产品约束

- 不改变现有 UI 的视觉语言、页面结构、布局、导航顺序、主题和已工作的交互流程。
- 不删除、重命名或重写已经工作的公开 API，除非保留兼容层并有回归测试。
- 修复状态所需的字段、文案和绑定属于必要修复，但不得借机重做 UI。
- 如果多实例确实需要新入口，优先复用现有 Models/Cluster 操作方式，以最小增量接入。
- UI 修改前保存基线截图，修改后使用相同尺寸、主题和状态数据做视觉回归。
- 不允许生产路径返回 skeleton、stub、mock 成功响应。
- 模型目录存在只代表 discovered，下载完成只代表 available，进程存在只代表 process_running；三者都不能代表 ready/loaded。
- `/v1/models` 返回名称不能作为权重已物化或能够生成首 token 的证据。
- 不通过无限 timeout 解决卡死。所有跨进程、跨节点和模型加载步骤必须有阶段、deadline、取消和明确错误。
- 不硬编码模型名称、固定端口或固定两节点规模；上述 A/B 拓扑是验证 fixture，不是架构上限。
- 先测量再优化。没有基准和正确性对照的优化不能进入默认路径。

## 开工前的实际调用链审计

确认稳定仓库中的实际 App bundle、启动脚本、Python 环境和源码映射，列出：

1. SwiftUI Models/Chat/MenuBar 到共享 TokenityStore 的状态来源。
2. TokenityStore 到 coordinator Node Agent 的 HTTP 调用。
3. coordinator 到 worker `/v1/node/start-distributed-rank` 等 typed HTTP 调用。
4. 每台 Agent 到本地 supervisor/rank process 的启动与停止。
5. rank 0 OpenAI API、非 0 rank generation loop 和 JACCL/Ring 数据面。
6. SSE 从 runtime、FastAPI、URLSession parser 到 SwiftUI 增量渲染的完整链路。
7. 系统、Node Agent、runtime/rank 与 UI 内存指标的来源。

当前仓库可能仍保留 legacy hostfile/`mlx.launch`/SSH helper 代码。必须区分“仍被生产调用”“仅 official experimental/测试兼容”“死代码”。先加调用链证据和测试，再安全弃用；任何旧 helper 都不得绕过 HTTP Node Agent 产品路径。

## 已知 Bug：逐项复现、定位、修复和验收

### Bug 1：模型加载后的第一次对话不是真正流式

不要先假定是模型没加载完。分别验证：

- 客户端是否发送 `stream: true`。
- HTTP 响应头是否立即提交且 Content-Type 为 `text/event-stream`。
- URLSession、SSE parser、FastAPI/uvicorn 或 UI 更新是否缓冲 chunk。
- runtime 是否在创建 StreamingResponse 之前同步加载 tokenizer、物化权重、prefill 或首次 Metal compile。
- 第一次请求是否承担 lazy materialization、图编译或 cache 初始化。
- UI 是否逐 chunk 更新主线程，还是等 `[DONE]` 后一次性写入。
- 当前“流式失败后自动非流式恢复”是否误把正常冷首 token 等待当成流式失败。

为每个请求记录 accepted、headers_sent、first_keepalive、prefill_start/end、first_content_token、last_token、cancelled/completed 时间。区分首个 SSE 字节、首个内容 token 和完整 TTFT。

修复要求：

- 必要的权重物化和受控 warmup 在 READY 之前完成。
- warmup 不污染聊天历史、统计或用户 KV cache；必须可取消、可超时。
- 长 prefill 期间发送合法 SSE comment/keep-alive；keep-alive 不能冒充内容 token 或 READY。
- 第一次 `stream=true` 对话也必须逐 chunk 到达客户端，用带时间戳的 `curl -N` 和 Swift parser 测试证明。
- 非流式 fallback 只能用于明确的传输失败，并保留原始错误；不得掩盖服务端没有流式输出的根因。
- 非流式 API 行为保持兼容。

### Bug 2：Models 显示 loaded，但模型没有真正加载成功

建立真实、按模型实例隔离的状态机，至少区分：

- discovered
- available/downloaded
- queued
- launching
- distributed_initializing
- loading_metadata
- materializing_weights
- compiling/warming
- ready
- busy
- unloading
- stopped
- failed

READY 必须基于：

- 目标实例及所有计划 rank 健康。
- 所有 rank 报告同一 instance/cluster id、model revision、world size、connection mode 和成功加载状态。
- 权重已物化或 runtime 提供等价的可靠完成证据。
- tokenizer/processor 与 generation engine 可用。
- 一 token readiness probe 或等价 warmup 成功。
- rank quorum 在 deadline 内完成；任一 rank 失败时不能显示绿色。

模型 inventory、OpenAI model catalog 和运行实例状态必须是不同 DTO。加载失败要保留根因、rank、时间、日志位置和 retry 状态。轮询错误或超时后不得保留旧绿灯。增加 generation thread death、端口占用、模型损坏、版本不一致和远程 rank 退出测试。

### Bug 3：双机 RDMA 正常，单机模式卡死

单机必须是一等执行路径：

- 只选择一台 Mac 时，通过该 Mac 的 Node Agent HTTP API 启动本机 runtime。
- 使用普通 `mlx_lm.load()` 或经过验证的单进程 provider。
- 不调用远程 rank endpoint、JACCL、分布式 hostfile、self-SSH 或不必要的 `mx.distributed.init()`。
- 不通过 SSH 启动单机 runtime。
- 单机和多机复用上层实例生命周期和 OpenAI API，但 execution plan 明确不同。
- 只有节点数大于 1 时才创建 distributed group。
- 每个初始化阶段有 deadline、watchdog 和结构化日志；超时后清理完整本地进程组并返回 FAILED。

至少验证：A 单机、B 单机、小模型冷启动、第一次/第二次流式、停止、重启、加载失败恢复；同时确保现有双机 HTTP + RDMA 路径不回退。

### Bug 4：菜单栏 Logo 与实际状态不符

菜单栏、主窗口、Models、Chat 和 Overview 必须使用同一个 TokenityStore/后端实例聚合状态，不能各自推导。

定义明确映射：offline/stopped、starting/loading/warming、ready、busy、degraded、failed。只有实例真实 READY、必需 Agent/rank 健康、world size 和 connection mode 与计划一致时显示绿色。轮询过期、Node Agent 不可达、部分 rank 失联或 topology 不一致时必须显示 degraded/failed，不得保留最后绿灯。

验证 App 启动、Node Agent 重启、后端重启、窗口关闭、睡眠唤醒、网络切换、模型卸载和活动流式生成。增加状态映射、共享 Store、轮询过期与旧响应覆盖测试。

### Bug 5：Overview 内存不是实际模型占用

不要混淆：系统 RAM、Tokenity/Agent 进程、runtime/rank phys_footprint、MLX Metal active/peak/cache、模型权重估算、mmap file cache、KV/prompt cache。

API/DTO 至少区分：

- system_total / system_used / available / pressure
- process_resident 或 macOS phys_footprint
- mlx_active / mlx_peak / mlx_cache（由对应 runtime 进程采集）
- model_weights_estimated / model_resident_observed
- file_cache/reclaimable（可可靠测量时）
- kv_cache / prompt_cache（可获得时）
- per-instance、per-rank、per-node 数值和 sampled_at

Apple Silicon 是统一内存，字段命名和解释必须准确。模型在 rank 子进程时，Node Agent 不能只报告自身 MLX 内存；应由 runtime/rank 上报进程内指标，Agent/控制面聚合。分布式权重按实际分片汇总，不把完整模型大小重复乘节点数。采样失败显示 unknown/stale。对比加载前、加载后、生成中、卸载后，并验证内存释放。

## 核心架构：单机、多机和多模型实例共存

不要继续用全局 `role -> process` 表示所有模型。建立显式 `ModelInstance`，至少包含：

- instance_id
- requested model id、resolved path/revision/tokenizer identity
- execution mode：single / ring / jaccl
- selected nodes、rank mapping、world size、parallel plan
- coordinator/gateway、内部端口、starting port
- process/rank identity、operation id、generation epoch
- lifecycle state/version、timestamps、deadline、heartbeat
- memory reservation、actual usage、active request count
- readiness evidence、last error、log paths

架构边界：

1. Node Agent 是长期存在的 HTTP 控制面代理。
2. 每个模型实例使用独立 runtime/rank 进程或进程组，隔离 MLX distributed 全局状态、Metal 生命周期、环境变量和崩溃。
3. supervisor 以 instance_id 管理进程，不再以笼统 role 唯一化；start/stop/restart 幂等。
4. 每个实例分配独立可回收的 HTTP/internal/collective 端口，启动前检测冲突。
5. 控制面维护节点资源账本：总内存、预留、实际占用、端口、RDMA capability、参与实例。
6. 加载前资源准入；失败或停止后释放 reservation，不能因过量加载拖死机器。
7. 稳定网关根据 model id/alias/instance id 路由；动态加载/卸载不要求重启网关。
8. 同一节点可参与多个实例，但必须不同进程并通过资源准入。例如 A+B 的 GLM RDMA 实例与 A 的 Qwen 单机实例并存。
9. 请求持有实例 lease；有在途请求时不得卸载，取消/断连后释放 lease、KV 引用和队列槽。
10. 同模型多实例有确定的负载均衡规则和可观测的路由结果。
11. 所有跨机器生命周期操作通过 Node Agent HTTP；不得用 SSH 实现任何一项。

先做纵向闭环：单机实例 → 动态加载/卸载 → 多机实例 → 两个实例共存 → 网关路由。每一步均有测试，避免一次性重写。

## 参考代码使用重点

### oMLX

重点阅读：

- `omlx/engine_pool.py`：EngineEntry、加载锁、lease、LRU、内存准入、失败回收和真实 loaded。
- `omlx/admin/routes.py`：模型 load/unload API、active models 和 memory/status DTO。
- `omlx/scheduler.py`、`omlx/engine/batched.py`、cache 模块：continuous batching、prefix/KV cache、取消和背压。
- Swift Models/MenuBar/SystemMetrics 相关文件：轮询生命周期和状态展示思路。

不要把 oMLX 单进程 EnginePool 假设直接套到跨节点实例，也不要复制 UI/Logo/资产。

### exo

重点阅读：

- `src/exo/shared/types/worker/instances.py`
- `src/exo/master/placement.py`、`placement_utils.py`
- `src/exo/worker/plan.py`
- runner supervisor、MLX builder/generator、info_gatherer

提炼实例/runner 边界、placement、warmup、流式和节点监控。exo 也可能有 TODO 和不适用假设；不要把 Zenoh/event sourcing/Rust 等无关复杂度整体引入。

## 性能优化：先基准、后决策

建立可重复 baseline，至少记录：

- 模型发现、冷加载、weight materialization、warmup 耗时
- headers latency、first SSE byte、TTFT
- prefill tokens/s、decode tokens/s
- inter-token latency p50/p95/p99
- 并发吞吐、排队和取消释放时间
- 单节点、双节点 Ring、双节点 JACCL 对比
- system/process/MLX/model/KV 峰值内存
- prefix cache 冷/热/部分命中

固定模型、prompt、sampling、max tokens，记录机器、OS、runtime Python、MLX/MLX-LM、连接拓扑和温度状态。包含冷/热启动、短/长 prompt、并发和多轮重复前缀。

只使用论文原文、官方 MLX/MLX-LM 文档和参考源码等第一手资料。评估并记录适用性、预期收益、正确性风险、内存代价、复杂度和采用决定：

- load 后 materialization + compile warmup
- persistent runtime
- MLX-LM continuous batching
- prefix/prompt cache、block/paged KV cache
- chunked prefill、decode-priority scheduling
- speculative decoding、Native MTP/Medusa/EAGLE 类方案
- KV cache quantization
- prefill/decode disaggregation
- JACCL/Ring topology、`MLX_METAL_FAST_SYNCH`、通信计算重叠
- 模型特定 Metal/custom kernel

阅读 PagedAttention/vLLM、Sarathi/Sarathi-Serve、DistServe/Splitwise、speculative decoding 等原始论文，但不得照搬 CUDA 假设。必须适配 Apple Silicon 统一内存、MLX lazy evaluation 和当前 MLX-LM API。

高风险优化放在 feature flag 后。优化前后用相同参数做 token/文本正确性对比，不接受 silent accuracy regression。收益低于噪声或稳定性下降时保留 baseline。

## 稳定性与鲁棒性要求

- start/stop/restart/load/unload 使用 operation id 和幂等语义。
- 状态有合法转移表和单调 generation/version，旧轮询不能覆盖新状态。
- rank 心跳包含 instance/cluster id、rank、world size、model revision、connection mode 和 epoch。
- 任一 rank 异常使实例 degraded/failed，停止新请求并清理其余 rank。
- Agent/网关重启后从 pid、端口和 runtime 握手重新对账；无法确认的进程标记 orphaned。
- 启动失败、模型损坏、磁盘不足、端口冲突、HTTP worker 失败、版本不一致、RDMA 断链和节点离线返回可操作错误。
- SSE 断开、Task cancellation 和 shutdown 向下传播并及时终止生成。
- 队列有最大长度、背压和公平性；长 prefill 不饿死 decode stream。
- 日志包含 instance/operation/request/rank/node id，不只依赖 fatal 字符串扫描。
- Stop 后验证所有本地/远程 rank 均由各自 Agent 清理，端口和 reservation 回收。
- 测试睡眠唤醒、网络恢复、Agent 短暂不可达和轮询乱序。

## 必须建立的测试矩阵

### 自动化测试

- 生命周期合法转移与 stale update 防护。
- 单机 plan 不包含 SSH、远程 rank、JACCL/hostfile 或 distributed init。
- 多机 plan 通过 typed HTTP rank API 编排且不包含 SSH。
- A/B 当前地址、RDMA device/IP 和版本 preflight。
- Mac B identity 不会回退到任何旧地址。
- supervisor 同时管理多个同类型、不同 instance_id runtime。
- 重复 start/stop、端口冲突、部分启动失败和 orphan cleanup。
- rank quorum、heartbeat timeout、generation thread death 和 worker crash。
- discovered/available/loading/ready/failed 不混淆。
- 第一次 stream 请求按时间增量收到 SSE 并结束 `[DONE]`。
- 非流式、stop sequence、usage、取消和客户端断开。
- 内存 DTO、聚合、stale/unknown 和卸载释放。
- 多实例资源准入、lease、路由和动态卸载。
- Swift Store/MenuBar/Models/Overview/Chat 状态一致、轮询取消和旧响应防护。
- 静态/单元测试保证生产代码没有 SSH 调用路径。

### 真实硬件验收

1. 验证 A/B SSH 仅用于 Codex 调试可达性；产品启动过程日志中没有 SSH。
2. 验证两台 Agent 的 runtime Python、mlx 0.31.2、mlx-lm 0.31.3+ 和 RDMA 拓扑。
3. A 单机加载小模型：READY 前不显示 loaded；第一次真流式；停止后内存回落。
4. B 单机同样验证。
5. 加载失败模型：Models、Overview、Chat、菜单栏均显示 failed，不出现绿色。
6. A+B Standard Network/Ring 加载并生成。
7. A+B JACCL/RDMA 加载目标 Qwen 并生成，记录 rank readiness、TTFT 和内存。
8. A+B 运行分布式 GLM，同时 A 运行单机 Qwen；统一 endpoint 正确路由并发请求。
9. 卸载 Qwen 不影响 GLM，再加载 Qwen 不重启 GLM/网关。
10. 生成中终止一个 worker rank：实例失败可见，无假 ready，无残留进程。
11. App/Agent 重启后状态重新对账，MenuBar/Models/Overview 一致。

为硬件验收提供脚本化命令、期望输出、超时和日志位置。SSH 命令只能作为 Codex 诊断步骤，不能出现在产品验收的控制链路中。

## 执行顺序

1. 确认稳定仓库、记录 dirty worktree、构建当前 UI。
2. 通过 HTTP 优先检查 A/B Agent、版本和 RDMA；SSH 仅在 HTTP 不足以诊断时使用。
3. 保存 UI 截图、API、日志和 benchmark 基线。
4. 逐项复现 5 个 Bug，输出带证据的根因表。
5. 审计并消除生产 SSH/legacy launcher 路径，保护当前 HTTP Node Agent 编排。
6. 实现真实状态机、rank quorum 和指标接口。
7. 修复单机执行计划与首次流式。
8. 将 supervisor/control plane 升级为 instance_id 级别。
9. 实现动态多实例、资源准入和统一网关路由。
10. 修复 MenuBar/Models/Overview/Chat 同源状态与内存展示。
11. 建立基准并实现经数据证明的低风险优化。
12. 完成故障注入、全量回归、Swift/Python 测试和真实硬件验收。
13. 最后审计死代码、资源泄漏、并发竞态、安全、日志和旧 Mac B 地址。

不要在审计后停下等待普通实现确认。只在缺少实际源码、硬件权限、必要凭据或会造成不可逆外部影响时请求输入。正常代码修改、测试、构建和本地诊断应自主推进。

## 构建与验证

- 使用项目支持的 Python 3.10+。
- 本地开发依赖来自稳定仓库的 pyproject；不要改变远程共享 runtime，除非该变更属于明确部署步骤。
- 运行 `./scripts/verify-stable-baseline.sh`。
- 运行完整 Python tests、Swift tests，并生成新的 app bundle。
- 用 `./scripts/run-tokenity-control-app.sh` 启动当前 UI。
- 不从旧 `MLX-Distributed/apps/TokenityControl` 启动。
- 对高风险改动先跑针对性测试，再跑全量回归。
- 不修改 oMLX/exo 参考仓库。

## 每阶段输出

每阶段报告：

1. 已复现行为与证据。
2. 根因，精确到文件、函数和状态/进程/HTTP 链路。
3. 修改内容及回归保护。
4. 测试/基准命令和结果摘要。
5. 使用过的 SSH 调试命令及原因；确认没有把 SSH 加入产品路径。
6. 尚未验证的硬件场景和下一步。

最终交付必须包含：

- 根因与修复对照表。
- HTTP 控制面、MLX 数据面、实例/网关架构图。
- 修改文件清单和现有用户改动保护说明。
- API/状态兼容说明。
- 自动化测试结果。
- 性能前后数据；无数据不得宣称提速。
- 单机、多机 RDMA、混合多实例验收记录。
- 版本/RDMA/address preflight 记录。
- 证明产品启动链路无 SSH 的日志或测试。
- 已知限制、回滚/feature flag 和剩余风险。
- UI 前后截图，证明无视觉回归且状态真实。

## 完成定义

只有同时满足以下条件才可以宣布完成：

- 所有工作、构建和启动来自 `/Users/zxc/Documents/Tokenity-Stable`。
- 从未启动旧 `MLX-Distributed/apps/TokenityControl` UI。
- Mac B 只使用 `192.168.5.75`（LAN）和 `192.168.0.2`（RDMA），没有恢复旧地址。
- 产品控制面只使用 Node Agent typed HTTP，数据面使用 MLX Ring/JACCL；没有 SSH 组网或 rank 编排。
- 第一次对话被时间证据证明为真流式。
- 加载失败不会显示 loaded/绿色。
- A/B 单机路径不再卡死且不触发分布式/SSH 路径。
- MenuBar、Models、Overview、Chat 对同一后端状态一致。
- 内存反映模型加载、生成和卸载的实际变化，并准确标注指标。
- 单机和多机实例可动态创建/卸载。
- 自动化环境证明多实例隔离、资源准入和路由正确；硬件结果诚实记录。
- A+B 分布式模型与 A 单机模型能够共存，不互相覆盖进程、端口、状态或路由。
- 现有功能和 UI 无回归。
- 全量测试通过，性能改进有可重复数据，失败路径无残留进程或虚假状态。
