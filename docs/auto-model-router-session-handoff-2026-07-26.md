# Tokenity 多模型常驻、自动路由与双机 GLM 修复交接记录

> 日期：2026-07-26
> 仓库：`HeyZhey/Tokenity`
> 开发分支：`feature/auto-model-router`
> 开发基线：`4594646a7d267a7c0e34e4c8d412d098540cea72`
> 唯一有效工作区：`/Users/zxc/Documents/Tokenity-Stable`

## 1. 文档边界

本文用于把本次开发会话中的项目基础信息、用户问题、可复核证据、工程判断、代码改动、远端部署、真实硬件验证、已知限制和后续操作一次性落盘。

安全边界：

- 不记录机器密码、私钥、Token 或其他凭据。
- 不记录不可验证的内部逐步推理；“工程判断”只保留观察、假设、验证结果、取舍和结论。
- SSH 仅是本次获得用户明确授权后的开发诊断和运维入口，不属于 Tokenity 产品控制面。
- 产品控制面仍然是 Node Agent typed HTTP，模型张量数据面仍然是 MLX Ring 或 JACCL/RDMA。

## 2. 本次会话的需求演进

本轮工作依次处理了以下诉求：

1. 先理解 Stable 项目的目录、架构、双机拓扑、测试和发布约束。
2. 实现多个模型常驻以及 `tokenity-auto` 本地自动选模。
3. 自测多个模型，验证是否能把不同策略的问题自动分配给不同模型。
4. 改善 Chat 美学；修复 Thunderbolt RDMA 状态被遮挡；把 Auto/手动选模和 Speed/Balanced/Quality 放到发送按钮附近，并采用滑块式策略选择。
5. 说明跨模型切换时的上下文语义。
6. 修复 “Auto routing unavailable, update legacy node agent” 和无法加载多个模型。
7. 修复 Resident Model Pool 中大量 Unavailable、Stop 无响应、单个 Allow Auto 导致全部卡片联动。
8. 修复 GLM-5.2 在双机环境中的加载和重复请求失败。
9. 修复 macOS 菜单栏显示警告三角形而不是 Tokenity 品牌 Logo。
10. 把当前代码与本轮工程信息提交并发布到 GitHub。

## 3. 不可违反的项目基线

### 3.1 正确源码与 UI

所有读取、修改、测试、构建和 UI 启动都来自：

```text
/Users/zxc/Documents/Tokenity-Stable
```

正确的 Swift App：

```text
/Users/zxc/Documents/Tokenity-Stable/apps/TokenityControl/.build/arm64-apple-macosx/debug/TokenityControl-Stable.app
```

正确 Bundle ID：

```text
ai.tokenity.control.stable
```

旧的 `/Users/zxc/Documents/MLX-Distributed` 只可作为只读参考，不能从那里构建或启动 Tokenity UI。

### 3.2 产品架构

```text
TokenityControl / OpenAI client
        |
        | typed HTTP :9100
        v
Coordinator Node Agent / gateway / Auto Router
        |
        | typed HTTP :9100
        v
Worker Node Agent
        |
        | local process supervision by (role, instance_id)
        v
MLX / MLX-LM ranks
        |
        | MLX Ring or JACCL/RDMA tensor traffic
        v
Apple Silicon shared-memory hardware
```

关键不变量：

- SwiftUI 不执行 SSH。
- Coordinator 不通过 SSH 编排 worker rank。
- 多实例身份是 `(role, instance_id)`，不能只按全局 role 管理。
- 普通客户端使用 Coordinator `:9100/v1`，rank-0 的 `:8000`/动态端口是私有 runtime。
- Loaded/Ready 必须来自实例身份、rank quorum、runtime readiness、warmup 和真实推理证据，不能由目录、进程或 `/v1/models` 名称替代。
- Stop 必须精确到 instance，不能误伤 sibling。
- 任何请求 lease 和共享生成槽都必须只释放一次。

## 4. 双机与运行环境

| 角色 | 主机 | LAN / Agent | Thunderbolt / RDMA | RDMA 设备 |
| --- | --- | --- | --- | --- |
| Coordinator | Mango | `192.168.5.23:9100` | `192.168.0.1` | `rdma_en4` |
| Worker | Kiwi | `192.168.5.75:9100` | `192.168.0.2` | `rdma_en5` |

共享安装路径：

| 用途 | 路径 |
| --- | --- |
| Python runtime | `/Users/Shared/TokenityRuntime/current/.venv/bin/python` |
| 正式 Tokenity 代码 | `/Users/Shared/TokenityCode` |
| 模型目录 | `/Users/Shared/TokenityModels` |
| Node Agent launchd label | `ai.tokenity.node-agent` |

本轮硬件验证时的核心版本：

- MLX：`0.32.0`
- MLX-LM：`0.31.3`
- 两台 Agent 最终代码 revision：`c620772a4c1fe7af37471dc06f467cab9d9df0fabf085b312d2569ca05d6e216`
- Node Agent 更新包：`dist/Tokenity-NodeAgent-0.1.0.pkg`

模型：

- `GLM-5.2-mxfp4`
- `Qwen3.5-4B-Native-MTP-4bit`

本轮没有把机器密码写入源码、日志文档或 Git。

## 5. 多模型上下文语义

结论：Chat 会话可以在模型切换后保留应用层上下文，但不能在不同模型之间迁移底层 KV cache。

- Tokenity 会保存会话的消息历史，并在后续请求中把所需消息发给实际选中的模型，因此用户看到的对话语义可以延续。
- Auto Router 保存会话粘性：首次选中模型后，后续追问默认保持同一 `model_id + revision`。
- 硬约束变化、模型失败、明显话题切换、上下文容量不足或用户明确要求时可以重新路由。
- 每条 assistant 消息保存实际模型、revision、instance、route reason、confidence 和 routing latency，切换后仍可追溯。
- 不同模型的 tokenizer、chat template 和 KV cache 不兼容；切换模型后需要重新 prefill 会话历史，不能继承上一模型的 GPU/MLX cache 状态。
- 一旦已经向用户输出 reasoning/content token，禁止无痕切换并拼接另一个模型的答案。自动 fallback 只允许发生在响应头/首 token 之前。

## 6. 根因、证据、决策与取舍

| 问题 | 观察和证据 | 根因 | 实施决策与取舍 |
| --- | --- | --- | --- |
| Auto routing unavailable | UI 能发现旧 Agent，但缺少多实例 contract/capability | Control UI 与远端 Agent 版本不一致；旧协议无法安全做精确实例操作 | 现代多实例必须同时具备 `managed_instances`、`instance_runtimes`、`instance_quorum`、`cluster_runtime`；不使用危险的 legacy stop-all 假装支持 |
| Resident Pool 大量 Unavailable | 已停止或远端已不存在的记录持续残留 | UI 仅追加本地记录，刷新时没有按 Agent 事实源恢复和剪枝 | 从 Agent instances/routes 恢复 Ready/Busy/Loading 实例；剪除 stopped、missing、不可恢复记录；长加载实例保持身份 |
| Stop 无响应 | 点击卡片 Stop 后实例未精确退出，或 404 被当作失败 | 旧路径依赖 active model/global cleanup，缺少卡片 instance ID | 每卡片调用精确 instance Stop；远端已经不存在时按幂等成功处理；禁止 global cleanup fallback |
| Allow Auto 全部联动 | 修改一张卡片会同时改变同模型其他卡片 | 状态按 model ID 建模，而不是按 instance ID | `allowsAuto` 按 instance ID 保存；请求通过 `allowed_instance_ids` 精确约束候选 |
| 长时间加载后 Unavailable | GLM 加载超过短心跳窗口，UI 丢失 active instance | UI 刷新覆盖加载身份；现代 Agent 拒绝 nil legacy heartbeat；短 TTL 可能缩短长启动 lease | UI 在加载期间只发带 instance ID 的 heartbeat；registry 使用 `max(existing_deadline, now + ttl)`；Coordinator 将实例 heartbeat 扇出到 worker ranks |
| GLM 第一次探针后再次失败 | 首次 load/probe 可成功，下一请求出现 `[jaccl] Recv failed with error code -12` | GLM-5.2 的直接 JACCL 请求循环在当前 MLX/JACCL 组合中残留跨请求 collective 状态 | 对 GLM 自动把 direct `jaccl` 提升为 `jaccl-ring`；牺牲直接模式，换取可重复请求稳定性 |
| GLM/JACCL 请求状态污染 | GLM 重复请求或 teardown 可能卡住 | MLX-LM BatchGenerator prompt cache、seed 同步和一元素控制 collective 在 JACCL 下保留/排队状态 | GLM/JACCL 使用 sequential generation；关闭 prompt cache；确定性 per-instance seed；控制帧改为多元素 CPU collective；GLM 跳过冗余 post-load barrier |
| GLM 索引器加载兼容 | 固定 MLX-LM 版本无法完整解释 checkpoint 的跨层 indexer 共享 | GLM-5.2 checkpoint 行为领先于已安装 MLX-LM | 采用与 MLX-LM PR #1410 对齐的兼容逻辑，不更改 checkpoint |
| Stop/状态轮询逐渐卡死 | 多次加载/停止后系统堆积不可中断探针进程 | JACCL teardown 后 `ibv_devinfo` 可能进入 macOS 内核 `U` 状态；轮询不断创建新阻塞任务 | `rdma_ctl` 已提供 authoritative port state 时不再额外运行 `ibv_devinfo`；已有内核阻塞只能通过节点重启清理 |
| Qwen 被误判 probe 失败 | SSE 已完成且生成 token，但 visible content/reasoning 为空 | 探针只看文本，不看 OpenAI usage | `usage.completion_tokens > 0` 也视为生成证据；仍要求流正常完成 |
| Auto allowed instance 返回 409 | 约束只允许 Qwen，但评分先选 GLM，之后才排除 | `allowed_instance_ids` 在模型评分后才应用 | 在生成 model profiles/runtime candidates 前过滤 instance；评分与 fallback 始终使用同一允许集合 |
| Fast 与 Quality 选到同一模型 | `4B` 模型没有被识别为快速小模型 | 参数规模推断缺少通用 `4B` 名称解析 | 增加 billion-scale 解析；Fast 偏向 Qwen 4B，Quality 偏向 GLM |
| Chat 顶部拥挤、RDMA 被遮挡 | 标题、状态、Auto、Balanced、操作按钮在同一横栏竞争宽度 | 信息层级和控制布局不合理 | 顶栏保留身份/状态；Auto/手动和策略滑块移入 composer；发送控件形成统一操作区 |
| 菜单栏显示红色警告三角 | 状态级别 SF Symbol 被直接当作主 Logo | 品牌身份和健康状态共用同一图形 | 主图标固定为 Tokenity 单色放射 Logo，状态只用小色点表达；保留 tooltip 和菜单内详细错误 |

## 7. Auto Router 实现

新增 `tokenity/control/routing.py`，保持纯逻辑、无网络、无进程副作用。

主要类型：

- `ModelCapabilityProfile`
- `ModelRuntimeState`
- `RouteContext`
- `RoutePolicy`
- `RouteDecision`
- `RouteReason`
- `CandidateScore`
- `RouterMetrics`
- `CapabilityRegistry`
- `AutoRouter`

路由顺序：

1. 用户显式模型/revision。
2. Ready、quorum、heartbeat、上下文长度、输出长度、tools、JSON、thinking、modality、语言、隐私、允许列表和参数兼容等硬约束。
3. 会话粘性。
4. coding、reasoning、long-context、tool-use、fast-chat、general 等确定性任务分类。
5. 质量、预测 TTFT、队列、失败率、会话亲和和用户优先级评分。
6. 低置信度时使用配置的默认模型。

策略：

- Fast：提高延迟和队列惩罚。
- Balanced：平衡质量、延迟、队列和失败率。
- Quality：提高质量权重并降低延迟惩罚。

API：

- `/v1/models` 返回 `tokenity-auto` 和 Ready 的实际模型。
- `/v1/router/decision` 只做决策，不启动推理。
- `/v1/chat/completions` 支持 `model: tokenity-auto`。
- Gateway 在选中实例后把 upstream `body.model` 重写成真实 `requested_model_id`。
- 返回 `X-Tokenity-Routed-Model`、`X-Tokenity-Instance-ID`、revision、reason、confidence、routing latency 等 provenance。
- 选中模型后才注入该模型的 sampling/chat-template 默认值；移除目标不支持的私有参数。
- 不记录原始 prompt，只保留派生特征和路由结果。

## 8. 多实例和共享硬件调度

`tokenity/control/instances.py` 的主要增强：

- 结构化内存 reservation：权重、KV/prompt cache、MLX cache、runtime 峰值和 OS headroom。
- 加载后使用 observed memory 调整 reservation，同时保护最低资源占用。
- request lease 阻止服务中的实例被卸载。
- heartbeat 不允许短调用缩短已有较长 lease。
- queue depth、active request count 和健康状态参与 instance 选择。
- 新增 overlap-node 共享生成槽：所选节点有交集的重型 MLX 实例默认 FIFO 串行；完全不重叠的实例可并行。
- 共享生成槽支持队列上限、取消、超时和幂等释放。
- collective port 分配检查整个候选端口范围，并排除已占用 HTTP port。

## 9. Node Agent 与 Gateway 改动

`tokenity/node_agent/agent.py` 的主要增强：

- 多实例 capability gate 和 instance-scoped start/stop/heartbeat/quorum。
- Coordinator instance heartbeat 扇出到所有 worker rank。
- 现代 managed instance 存在时拒绝无 instance ID 的 legacy global heartbeat。
- `/v1/models`、`/v1/router/decision`、`/v1/chat/completions` 接入 Auto Router。
- `allowed_instance_ids` 在构建候选时过滤，而不是评分后过滤。
- exact instance 和 Auto 请求都重写 upstream model。
- 按实际 selected profile 过滤/注入 runtime 和 chat-template 参数。
- 首 token 前允许兼容 fallback；流读取开始后不再换模型。
- 请求 lease 和 generation slot 使用统一的幂等清理路径，覆盖连接失败、响应启动失败、取消和断连。
- 队列满返回 `429` 与 `Retry-After`。
- 路由耗时只统计本地决策，不把 upstream connect time 混入。
- GLM direct JACCL 自动降级到已验证稳定的 `jaccl-ring`。
- Qwen 4B 参数规模识别和 Fast/Quality 模型差异化。

## 10. MLX / MLX-LM / RDMA 改动

`tokenity/serving/distributed_openai.py`：

- GLM/JACCL 使用 sequential generation，避免 BatchGenerator 跨请求状态。
- GLM/JACCL 所有 rank 使用 `LRUPromptCache(0)`；warmup 不重新开启。
- GLM 跳过重复的 post-load JACCL barrier。
- 为 MLX-LM ResponseGenerator 安装确定性 per-instance seed，避免在请求循环里使用不稳定的一元素 seed collective。
- JACCL 控制对象采用多元素 CPU frame，模型 collective 保持不变。
- 保持 GLM 的进程隔离 teardown 语义，避免 CompilerCache 析构问题。
- 保留 GLM-5.2 跨层 indexer 兼容。

`tokenity/mlx/rdma_probe.py`：

- `rdma_ctl` 已经报告端口状态时不再调用 `ibv_devinfo`。
- `rdma_ctl` 缺失 authoritative state 时仍保留 `ibv_devinfo` fallback。
- 目标是避免状态刷新制造不可中断的内核探针堆积。

## 11. SwiftUI 改动

### 11.1 Chat

- Auto/手动模型选择器移到 composer。
- Speed/Balanced/Quality 使用三段连续滑块。
- 支持本会话模型锁定。
- 路由阶段单独展示，不伪装成模型 token。
- Assistant 消息保存并展示实际 routed model、revision、instance、reason、confidence 和 latency。
- Swift stream transport 暴露 HTTP response metadata，不再丢弃 route headers。
- Auto 请求只发送通用字段、策略、session 和允许实例约束；手动模式继续使用所选模型专属生成配置。
- 重新组织标题、状态和 composer 的视觉层级，修复 Thunderbolt RDMA 文案被遮挡。

### 11.2 Resident Model Pool

- 从 Agent instances/routes 恢复模型池。
- 按 instance ID 建模和显示 Ready/Busy/Queue、topology、内存、TTFT 和 capabilities。
- 每张卡片独立 Keep Resident / Allow Auto / Use in Chat / Stop。
- 远端 missing/stopped 实例被剪枝。
- 长加载实例不会因后台刷新丢失。
- 加载第二个模型不会无条件取消第一个模型正在进行的 Chat。
- 非活跃 sibling 的 quorum 异常不会污染当前健康实例。

### 11.3 Overview 与 MenuBar

- 汇总多模型 Ready/Busy 与 Auto routing health。
- MenuBar 与主窗口继续共享 AppDelegate 持有的唯一 `TokenityStore`。
- 主菜单栏图标改为 Tokenity 单色放射 Logo；健康状态改用右下角小色点：
  - Ready：绿色
  - Busy：黄色
  - Warning：红色
  - Stopped：灰色
- Logo 提供 Light/Dark visual smoke test。

## 12. 脚本与基准

- `scripts/benchmark-openai-stream.py`
  - 支持多场景、路由矩阵、route headers、TTFT、queue wait、吞吐和结果输出。
  - 保持真实硬件不可用时不伪造通过。
- `scripts/benchmark-auto-router-matrix.example.json`
  - 提供 Auto Router 多策略测试样例。
- `scripts/preflight-tokenity-cluster.py`
  - 检查 Agent revision、capabilities、RDMA/address 和环境一致性。
- `scripts/package-tokenity-agent-update.sh`
  - 生成代码型 Node Agent 更新 pkg。
  - 安装路径与 launchd 生命周期固定，不把 SSH 写入产品。

## 13. 测试覆盖

Python 回归覆盖：

- 纯路由硬约束、revision-scoped sticky、Fast/Balanced/Quality、低置信度默认模型和规则路由 p95。
- exact instance model rewrite、Auto gateway、route headers、允许实例、profile 参数过滤。
- 首 token 前 fallback、流开始后禁止 fallback、lease/slot 只释放一次。
- 队列计数、429、共享节点串行、非重叠节点并行、取消和超时。
- capability gate、worker heartbeat、长 lease、collective port、内存 reservation。
- GLM jaccl-ring fallback、sequential engine、prompt cache、seed、control collective 和 barrier。
- RDMA 探针 authoritative state/fallback。

Swift 回归覆盖：

- UI 重启恢复 resident instances。
- 长加载身份和 heartbeat。
- 第二模型 capability gate。
- sibling 加载不取消 Chat。
- sibling quorum 隔离。
- Auto/manual 请求和 provenance。
- per-instance Allow Auto。
- 精确 Stop 和 missing 幂等 Stop。
- stopped/missing resident 剪枝。
- Chat/Resident Pool/MenuBar Light/Dark visual smoke。
- 正确品牌菜单栏 Logo。

本轮发布前最后一次完整结果：

```text
Python: 142 passed, 8 skipped
Swift:  101 tests, 0 failures
```

完整 Stable App 构建成功，Bundle ID 校验为 `ai.tokenity.control.stable`。

## 14. 真实双机验证

两模型同时 Ready 时：

- `/v1/models` 同时列出 `tokenity-auto`、GLM 和 Qwen。
- 两个实际实例均为 2/2 quorum、`Ready`、`health_ready=true`。
- 两模型均使用 `jaccl-ring`。
- 没有 orphan rank。

真实 Auto Router 结果：

| 请求 | 实际模型 | 响应 | Route reason | Confidence | Routing latency | Queue wait |
| --- | --- | --- | --- | --- | --- | --- |
| `fast` | Qwen 4B | `AUTO FAST OK` | `scored_best` | `1.000` | `0.052 ms` | `0.059 ms` |
| `quality` | GLM-5.2 | `AUTO QUALITY OK` | `scored_best` | `0.750` | `0.039 ms` | `0.040 ms` |

补充证据：

- GLM 请求报告 `cached_tokens=0`，符合关闭分布式 prompt cache 的设计。
- 强制 `allowed_instance_ids` 后，两模型也都能被精确选择。
- GLM direct `jaccl` dry-run/start 会自动返回 `jaccl-ring` rank plan。
- 本轮结束前为避免无谓占用，两个模型实例均已停止；这不否定上述同驻和路由验证。

## 15. 正确 UI 验证

构建和启动均来自 Stable 仓库：

```bash
cd /Users/zxc/Documents/Tokenity-Stable
./scripts/build-tokenity-control-app.sh
./scripts/run-tokenity-control-app.sh
```

最后验证的 App：

```text
/Users/zxc/Documents/Tokenity-Stable/apps/TokenityControl/.build/arm64-apple-macosx/debug/TokenityControl-Stable.app
```

菜单栏 Light/Dark 渲染证据：

```text
/tmp/tokenity-menubar-logo-light.png
/tmp/tokenity-menubar-logo-dark.png
```

两个渲染都显示 Tokenity 放射品牌图形和独立状态点，不再以警告三角形作为主 Logo。

## 16. 工程判断记录

### 16.1 为什么不使用另一个大模型做 Router

本地规则路由在真实请求中的决策耗时约为百分之一毫秒量级。调用生成式 Router LLM 会增加 TTFT、内存和失败面，且当前 Fast/Quality 需求可以由能力画像、硬约束和可解释评分解决，因此 MVP 保持纯规则。

### 16.2 为什么共享节点上的大型模型默认串行生成

Apple Silicon 统一内存能同时常驻多个模型，但不代表共享 GPU 可以无损并发 prefill/decode。节点有交集的实例先共享一个 FIFO 生成槽，优先保证可预测 TTFT、避免内存峰值和 collective 争用；未来只有真实基准证明安全才放宽。

### 16.3 为什么 GLM 自动选择 jaccl-ring

direct JACCL 能完成加载和首个探针，却在后续请求稳定复现 receive `-12`。这是“首次成功、重复请求不可靠”的更危险状态。自动选择已经通过连续请求验证的 jaccl-ring，比暴露一个表面更直接但不稳定的选项更符合 Ready 语义。

### 16.4 为什么菜单栏 Logo 与状态分离

Logo 表达产品身份，状态图标表达运行状态。用红色警告三角替换品牌图形会让用户误以为应用本身丢失 Logo。固定品牌轮廓、用小色点表达健康，同时在 tooltip/menu 中保留完整错误，信息更稳定。

### 16.5 为什么不保留不可用 Resident 记录

Resident Pool 是运行事实，不是历史日志。远端 stopped/missing 实例继续显示为 Unavailable 会制造幽灵对象、破坏 Stop/Allow Auto 语义。历史应由 Logs/Chat 保存，Pool 只保留可恢复的 Loading、Ready、Busy 和明确故障实例。

## 17. 已知限制与剩余风险

1. macOS/JACCL 异常 teardown 后，既有 `ibv_devinfo` 可能已经进入不可中断内核状态；代码能阻止继续制造新任务，但清理既有状态仍可能需要重启节点。
2. 不同模型切换不能迁移 KV cache，因此长会话切换会重新 prefill。
3. Auto Router 当前是规则/画像 MVP，没有 embedding 或小分类器；这是刻意的低延迟选择。
4. 质量分数和 TTFT profile 需要持续由真实 benchmark 更新；revision 改变后不能沿用旧画像。
5. GLM direct JACCL 当前被主动提升到 jaccl-ring；只有底层 MLX/JACCL 修复并完成连续请求回归后才应重新开放。
6. 真实并发性能需要按短/中/长 prompt 和 memory pressure 持续记录；“可同时常驻”不等同于“可无损同时生成”。
7. `/tmp` visual smoke 图片不是长期构建产物；长期证据应由 CI artifact 或发布截图保存。

## 18. 关键回滚点

- UI Auto 模式可切回手动模型选择。
- 每个 resident instance 可以独立关闭 Allow Auto。
- GLM 连接模式 fallback 集中在 `_stable_connection_mode_for_model`，便于底层修复后单点调整。
- Auto Router 是独立纯逻辑模块；手动 model gateway 保持兼容。
- 共享生成槽是 Gateway 调度层，不改变模型权重和 checkpoint。
- Node Agent 更新使用 pkg/launchd，可通过安装上一版本包回退。

## 19. 后续开工检查

```bash
cd /Users/zxc/Documents/Tokenity-Stable
git status -sb
git diff --check
./scripts/verify-stable-baseline.sh
```

远端检查优先通过 Agent HTTP：

```bash
curl --noproxy '*' -sS http://192.168.5.23:9100/v1/node/info
curl --noproxy '*' -sS http://192.168.5.75:9100/v1/node/info
curl --noproxy '*' -sS http://192.168.5.23:9100/v1/models
```

开始真实模型测试前再次确认：

- 两台 Agent code revision 一致。
- 四项 managed-instance capabilities 全部存在。
- RDMA port active，Thunderbolt 地址正确。
- 没有旧 UI 或旧 Bundle ID 进程。
- 没有遗留 orphan rank 或不可中断探针持续增长。
- 模型 load lease 不会被短 UI heartbeat 缩短。
- `tokenity-auto` 的 Fast 与 Quality 决策仍分别指向预期模型。

## 20. 本轮修改文件分组

### SwiftUI

- `apps/TokenityControl/Sources/TokenityControl/ChatViews.swift`
- `apps/TokenityControl/Sources/TokenityControl/MenuBar.swift`
- `apps/TokenityControl/Sources/TokenityControl/Models.swift`
- `apps/TokenityControl/Sources/TokenityControl/TokenityStore.swift`
- `apps/TokenityControl/Sources/TokenityControl/Views.swift`
- `apps/TokenityControl/Tests/TokenityControlTests/ChatExperienceTests.swift`
- `apps/TokenityControl/Tests/TokenityControlTests/ChatVisualSmokeTests.swift`
- `apps/TokenityControl/Tests/TokenityControlTests/MenuBarTests.swift`
- `apps/TokenityControl/Tests/TokenityControlTests/ResidentModelRecoveryTests.swift`
- `apps/TokenityControl/Tests/TokenityControlTests/TokenityStoreTests.swift`

### Python 控制面、数据面与路由

- `tokenity/control/__init__.py`
- `tokenity/control/instances.py`
- `tokenity/control/routing.py`
- `tokenity/mlx/rdma_probe.py`
- `tokenity/node_agent/agent.py`
- `tokenity/serving/distributed_openai.py`

### 脚本

- `scripts/benchmark-openai-stream.py`
- `scripts/benchmark-auto-router-matrix.example.json`
- `scripts/package-tokenity-agent-update.sh`
- `scripts/preflight-tokenity-cluster.py`

### Python tests

- `tests/test_benchmark_script.py`
- `tests/test_distributed_openai.py`
- `tests/test_instances.py`
- `tests/test_node_agent.py`
- `tests/test_preflight_script.py`
- `tests/test_rdma_probe.py`
- `tests/test_routing.py`

### 基础和审计文档

- `docs/development-prior-knowledge.md`
- `docs/stability-audit-2026-07-17.md`
- `docs/tokenity-stability-performance-multi-instance-session-prompt.md`
- `docs/auto-model-router-session-handoff-2026-07-26.md`
