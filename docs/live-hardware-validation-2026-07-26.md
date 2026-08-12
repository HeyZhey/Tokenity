# Tokenity 真实硬件首轮完整验证报告 — 2026-07-26

## 1. 报告结论

本轮在两台真实 512 GiB Apple Silicon Mac 上，使用生产 Node Agent、Gateway、
MLX/JACCL 和 Thunderbolt RDMA 路径完成了单机推理、单机路由、独立多机推理及
多模型路由验证。测试不以端口连通或单次 smoke request 作为通过依据，而是加入了
短中长输出、长上下文、连续请求、并发排队、SSE、客户端取消、恢复请求、会话粘性、
模型切换、停止、重载和资源观测。

总体结论：

- 单机 4B、单机 122B、独立多机 4B 和独立多机 122B 的基础推理链路可用。
- 单机双模型路由的选模、粘性、切换、排队、SSE 和取消恢复可用。
- 多机 GLM-5.2 只能短时工作，重复请求后出现 JACCL `-12` 并崩溃，判定失败。
- 两个模型同时采用双机分布式拓扑时，第二个 122B 卡在加载完成后的分布式同步，
  未达到 ready，判定失败。
- 由于 GLM 卸载后 Mac B 存在严重 wired 内存残留，本轮没有继续扩大危险压力，
  而是采用“双机分布式 4B + Mac A 单机 122B”的混合拓扑完成路由验证。该替代
  拓扑的功能、交替切换、并发和 SSE 测试成功，但不能代替“两个模型均为双机
  分布式实例”的验收。
- 发现 5 个 P0 和多项 P1 问题。当前版本不满足完整真实硬件发布门槛。

### 1.1 状态定义

- **通过**：目标拓扑和目标能力均有真实硬件证据，测试结束后仍健康。
- **部分通过**：基础链路通过，但正确性、API 合约、生命周期或清理存在缺陷。
- **失败**：出现崩溃、5xx、无法 ready，或目标拓扑无法完成。
- **因安全边界采用替代拓扑**：原目标继续测试会放大机器风险，改用更安全的拓扑
  验证上层能力；替代结果不得记为原目标通过。

### 1.2 验收矩阵

| 验证域 | 状态 | 首轮结论 |
|---|---|---|
| 双机环境与 RDMA 预检 | 通过 | 两端版本、模型 revision 和 RDMA 均一致且无预检问题 |
| 单机推理 | 部分通过 | Mac A 的 4B/122B、Mac B 的 4B 可稳定推理；严格输出、工具和部分长上下文不通过 |
| 单机路由 | 部分通过 | 精确路由、Auto、粘性、切换、并发、SSE、取消恢复通过；API 合约和上下文能力判断失败 |
| 独立多机 4B | 部分通过 | 2/2、基准、连续请求和取消恢复通过；45K 严格输出失败，第二次加载失败 |
| 独立多机 122B | 部分通过 | 2/2、基准、连续请求、18K 上下文和取消恢复通过；工具与语义正确性失败 |
| 独立多机 GLM-5.2 | 失败 | 初始请求成功，随后 JACCL `-12`、协调进程 `-6`、恢复请求 502 |
| 多机路由：4B 与 122B 均双机分布式 | 失败 | 4B ready；第二个 122B 长时间停在 148/148 后的同步阶段，始终未 ready |
| 多机路由：分布式 4B + 单机 122B | 因安全边界采用替代拓扑；成功 | 8 个功能用例、10 次交替、并发 2/4、Auto SSE 均成功 |
| 模型停止与资源清理 | 部分通过 | 最终活动实例、预留、孤儿进程均为 0；但 GLM wired 残留严重 |

## 2. 测试环境

证据根目录：

```text
dist/validation-2026-07-26/
```

### 2.1 节点与软件

| 项目 | Mac A | Mac B |
|---|---|---|
| Node Agent | `<node-a-lan-ip>:9100` | `<node-b-lan-ip>:9100` |
| 统一内存 | 512 GiB | 512 GiB |
| Thunderbolt IP | `<node-a-rdma-ip>` | `<node-b-rdma-ip>` |
| RDMA | `rdma_en4` active | `rdma_en5` active |
| MLX | 0.32.0 | 0.32.0 |
| mlx-lm | 0.31.3 | 0.31.3 |
| Tokenity | 0.1.0 | 0.1.0 |
| 代码 revision | `c620772a…6e216` | `c620772a…6e216` |

预检文件 `preflight.json` 显示 `ok: true`、`cluster_issues: []`，两端 Python
均来自 `${TOKENITY_RUNTIME_PYTHON}`。

### 2.2 模型

| 模型 | Revision | Inventory 大小 |
|---|---|---:|
| Qwen3.5-4B-Native-MTP-4bit | `aae0029b…cebbf6` | 约 3.12 GB |
| Qwen3.5-122B-A10B-4bit | `2cfc5d6b…a3ed` | 69.62 GB |
| GLM-5.2-mxfp4 | `411bb8dd…b0ce0` | 约 395.11 GB |

本轮启动参数中的 Native MTP 均为 `off`，因此本报告不构成 Native MTP
功能或性能验收。

### 2.3 判定原则

本报告把以下三类结果分开：

1. **传输/运行成功**：HTTP 200、SSE 完整、进程和 quorum 健康。
2. **路由成功**：实际 model、instance、revision 和 route reason 符合预期。
3. **内容/合约成功**：数学答案、JSON 语义、工具调用、token 上限和严格格式正确。

例如，HTTP 200 但算错答案只算运行成功，不算内容正确；长上下文找到了全部 marker
但多输出标签，只算信息保留成功，不算严格匹配成功。

本轮数学基准的正确答案为：

```text
987654321 × 37 - 123456789 = 36419753088
```

`80235` 的正确质因数分解为：

```json
[3, 3, 5, 1783]
```

## 3. 单机推理

### 3.1 Mac A：Qwen 4B

**状态：部分通过。**

- 短、中、长输出基准均无 HTTP 失败。
- 解码速度约 167.4–176.1 tok/s。
- TTFT 均值约 0.216–0.255 秒。
- 11,299 token 和 42,099 token needle 均严格返回
  `ALPHA|BETA|GAMMA`。
- 数学请求未返回正确答案；无 thinking 时返回 `36529876540`。
- JSON、工具调用和仅输出最终答案等严格合约未通过。

主要证据：

- `single-4b-short-benchmark.json`
- `single-4b-medium-benchmark.json`
- `single-4b-long-benchmark.json`
- `single-4b-needle-800.json`
- `single-4b-needle-3000.json`
- `single-4b-math-no-thinking.json`
- `single-4b-json.json`
- `single-4b-tools.json`

### 3.2 Mac A：Qwen 122B

**状态：部分通过。**

- 短、中、长输出共 9 个基准请求，全部成功。
- 解码速度约 59.4–60.3 tok/s。
- TTFT 均值约 0.418–0.481 秒；短请求首轮有冷态波动。
- 数学结果 `36542819700` 错误。
- JSON 可解析，但把非素数 `5349` 当作 prime factor。
- 强制天气工具调用时没有返回 `tool_calls`。

主要证据：

- `single-122b-direct-short.json`
- `single-122b-direct-medium.json`
- `single-122b-direct-long.json`
- `single-122b-math.json`
- `single-122b-json.json`
- `single-122b-tools.json`

### 3.3 Mac B：Qwen 4B

**状态：部分通过。**

- 改用可用端口后，短输出 5/5 成功，约 169.4 tok/s，TTFT 均值约
  0.189 秒。
- 连续请求 20/20 成功，平均 0.216 秒，p95 约 0.370 秒。
- 90,075 token needle 返回 HTTP 200 且三个 marker 全部存在，但添加了
  `FIRST/MIDDLE/LAST SECRET MARKER` 标签，严格匹配失败。
- 初始端口/Agent 环境受到重复 LaunchAgent 干扰，说明首次安装或升级后的服务
  去重仍需验证。

主要证据：

- `single-b-4b-direct-short.json`
- `single-b-4b-complex.json`
- `single-b-4b-port1-ready.json`
- `single-b-4b-port1-stop.json`
- `mac-b-after-port8000-failure.json`
- `mac-b-duplicate-launchagent.txt`

### 3.4 单机未覆盖项

- 没有在单台机器上加载 GLM-5.2；在后续 GLM 多机测试产生严重 wired 残留后，
  继续尝试单机 GLM 不符合安全边界。
- 没有在 Mac B 上重复完整的单机 122B 矩阵。
- Native MTP 未开启。

## 4. 单机双模型路由

**状态：部分通过。**

拓扑为 Mac A 上同时常驻单机 4B 与单机 122B，通过 Node Agent 的 Gateway
访问。

### 4.1 已通过

- 显式指定 4B 和 122B 均路由到正确 instance。
- Fast 选择 4B，Quality 选择 122B。
- `allowed_instance_ids` 能把 Fast 限制到 122B、把 Quality 限制到 4B。
- 同一 session 保持 4B；声明 `topic_changed` 后切换到 122B；锁定后继续保持
  122B。
- 空 allowed list、锁定冲突和不支持的 image modality 均返回 409，而不是静默
  选错模型。
- 20 次显式模型交替和 20 次 Fast/Quality Auto 交替全部 HTTP 200，并路由到
  预期模型。
- Router 自身决策延迟均在亚毫秒量级。

证据：

- `single-routing-functional.json`
- `single-routing-alternating.json`
- `single-decision-fast.json`
- `single-decision-quality.json`
- `single-decision-allowed-4b.json`
- `single-decision-explicit-122b.json`
- `single-routing-lifecycle.json`

### 4.2 并发与共享调度

| 并发 | 成功 | 观测最大 active | 观测最大 queue | 最大 queue wait | 最长总耗时 |
|---:|---:|---:|---:|---:|---:|
| 2 | 2/2 | 1 | 1 | 1,367 ms | 4.77 s |
| 4 | 4/4 | 1 | 3 | 6,127 ms | 9.97 s |

这证明重叠节点上的不同实例共用一个生成槽并串行排队；没有死锁或请求丢失，但并发
增加主要转化为排队时间，并非吞吐线性增长。

证据：

- `single-routing-concurrency-2.json`
- `single-routing-concurrency-4.json`

### 4.3 SSE、取消与恢复

- Auto SSE 返回 `text/event-stream`。
- `[DONE]` 恰好一次，无 malformed event。
- 客户端取消后 active/queued 在 10 秒内回到 0。
- 后续恢复请求返回 HTTP 200 和 `RECOVERED`。
- 最终两个实例均为 ready、active 0、queued 0。

证据：`single-routing-stream-edges.json`。

### 4.4 未通过的 API 合约

- 请求 `max_completion_tokens: 3`，实际返回 51 completion tokens。
- 请求 `logprobs: true` 和 `top_logprobs: 2`，响应仍为 `logprobs: null`。
- 运行实例宣称 tools capability，但强制工具调用仍返回 `tool_calls: null`。
- Router 对 42,099 token 输入把 4B、122B 都判为 `context_capacity`，返回
  `no_eligible_model`；同一轮 4B 直连已经严格处理 42,099 token。

证据：

- `single-routing-functional.json`
- `single-decision-context-42099.json`
- `single-4b-needle-3000.json`

## 5. 独立多机推理

### 5.1 Qwen 4B

**状态：部分通过。**

- 双机 JACCL ring 达到 2/2 quorum。
- 短、中、长基准共 10/10 成功。
- 连续复杂请求 20/20 成功，平均延迟 0.281 秒，最大 0.387 秒。
- 客户端取消后 10 秒内释放；恢复请求 HTTP 200；最终 quorum ready。
- 45,071 token needle 的三个 marker 全部存在，但带额外标签，严格匹配失败。
- 数学结果 `30259641879` 错误。
- JSON 中 `[3, 5, 5349]` 的乘积正确，但 `5349` 不是素数，不满足 prime
  factors 合约。
- 工具调用为空。
- 正常停止成功，A/B 均无 live instance 或 orphan。

首次实例成功不代表重载成功：同一 4B 停止后第二次加载时，Mac B generation
线程在 `_share_object` 中发生：

```text
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8d ...
```

worker 停止、quorum 不 ready、Gateway 返回 409，判定为生命周期失败。

主要证据：

- `multi-4b-quorum-initial.json`
- `multi-4b-direct-short.json`
- `multi-4b-direct-medium.json`
- `multi-4b-direct-long.json`
- `multi-4b-complex.json`
- `multi-4b-stop.json`
- `multi-4b-after-stop-a.json`
- `multi-4b-after-stop-b.json`
- `multi-4b-reload-quorum-ready.json`
- `multi-4b-reload-chat-ready.headers`
- `multi-4b-reload-stop.json`

### 5.2 Qwen 122B

**状态：部分通过。**

- 双机达到 2/2 quorum。
- 短、中、长基准共 6/6 成功。
- 连续复杂请求 10/10 成功，平均延迟 0.446 秒，最大 0.545 秒。
- 18,071 token needle 严格匹配成功。
- 客户端取消后释放；恢复请求 HTTP 200；最终 quorum ready。
- 数学结果 `36542714808` 错误。
- JSON factors `[3, 5, 17, 314]` 既含非素数，乘积也不等于 80235。
- 工具调用为空。
- 停止命令最终使两端实例停止。

主要证据：

- `multi-122b-quorum-ready.json`
- `multi-122b-direct-short.json`
- `multi-122b-direct-medium.json`
- `multi-122b-direct-long.json`
- `multi-122b-complex.json`
- `multi-122b-stop.json`

### 5.3 GLM-5.2

**状态：失败。**

成功阶段：

- 初始达到 2/2 quorum。
- 短、中、长基准共 6/6 成功。
- 连续请求 12/12 成功。
- 8,062 token needle 严格匹配成功。
- 客户端取消后 active/queued 曾正常释放。

失败阶段：

- 取消后的恢复请求返回 502。
- 协调进程因未捕获的 JACCL 异常退出，return code 为 `-6`：

```text
libc++abi: terminating due to uncaught exception ...
[jaccl] Send failed with error code -12
```

- 最终 quorum 不再 ready。
- 数学和 JSON 语义也不正确，工具调用为空。

主要证据：

- `multi-glm-quorum-3.json`
- `multi-glm-direct-short.json`
- `multi-glm-direct-medium.json`
- `multi-glm-direct-long.json`
- `multi-glm-complex.json`
- `multi-glm-stop-after-cancel-failure.json`

## 6. 多模型多机路由

### 6.1 目标拓扑：4B 与 122B 均为双机分布式实例

**状态：失败。**

执行顺序：

1. 启动双机分布式 4B，达到 2/2 ready。
2. 在 4B 保持运行时启动第二个双机分布式 122B。
3. 两端都完成 148/148 参数 materialization，但长时间停在
   `Loading model parameters (148/148)` / post-load distributed
   `all_sum` 附近。
4. 连续五次 quorum 快照均为 `ready: false`，因此没有进入路由请求阶段。
5. 在 Mac B 内存压力继续恶化前主动停止卡住的 122B。
6. 停止 122B 后，原 4B 仍为 2/2 ready，heartbeat 正常，证明 sibling
   精确停止没有把健康实例一起终止。

该结果是目标拓扑失败，不是“仅启动较慢”。122B 已经报告约 69.57 GB observed
aggregate，但 readiness evidence 仍缺失，等待数分钟仍无进展。

证据：

- `route-multi-4b-start.json`
- `route-multi-4b-quorum.json`
- `route-multi-122b-start.json`
- `route-multi-122b-quorum-1.json`
- `route-multi-122b-quorum-2.json`
- `route-multi-122b-quorum-3.json`
- `route-multi-122b-quorum-4.json`
- `route-multi-122b-quorum-5.json`
- `route-multi-122b-loading-a.json`
- `route-multi-122b-loading-b.json`
- `route-multi-122b-loading-stop.json`
- `route-multi-4b-after-sibling-stop-quorum.json`
- `route-multi-4b-after-sibling-stop-heartbeat.json`

### 6.2 安全替代拓扑

**状态：因安全边界采用替代拓扑；功能成功。**

拓扑：

- 4B：Mac A + Mac B，双机分布式 JACCL ring，instance
  `route-multi-4b-20260726`。
- 122B：仅 Mac A 单机，instance `route-single-122b-20260726`。

采用该拓扑的原因不是为了降低验收标准，而是纯双分布式尝试期间 Mac B
`pressure_available_ratio` 已降至 0.13–0.18，继续同时运行两个 JACCL
模型并施加长上下文/并发压力不安全。

结果：

- 初始两个 route 都为 ready。
- Router decision：
  - Fast → 分布式 4B；
  - Balanced → 分布式 4B；
  - Quality → 单机 122B。
- 8 个功能用例全部 HTTP 200，包括：
  - 两个实例的显式路由；
  - Fast/Quality；
  - session sticky；
  - topic change；
  - model lock；
  - allowed instance 限制。
- 10 次 Auto 交替全部成功：
  - 5 次 Fast → 分布式 4B；
  - 5 次 Quality → 单机 122B。
- Auto SSE 返回 HTTP 200、`[DONE]` 一次、无 malformed event，并路由到
  分布式 4B。

并发结果：

| 并发 | 成功 | 最大 active | 最大 queue | 最大 queue wait | 最长总耗时 |
|---:|---:|---:|---:|---:|---:|
| 2 | 2/2 | 1 | 1 | 1,836 ms | 2.84 s |
| 4 | 4/4 | 1 | 3 | 3,621 ms | 5.56 s |

混合拓扑同样受共享生成槽约束，但所有请求均完成，且 route provenance 正确。

限制：

- 这不是“4B 与 122B 均双机分布式”的通过证据。
- 最终采样时分布式 4B 处于瞬时 `prefill_pending`，`health_ready: false`，
  `final_routes` 暂时只列出单机 122B；此前 heartbeat 和所有请求成功。该瞬时
  健康/路由表抖动仍需作为生命周期回归项。
- 两个实例随后通过各自的 instance-scoped stop endpoint 精确停止；最终 A/B
  的活动实例、内存/端口预留、孤儿进程和模型进程均为 0。Mac B 的 wired
  仍高达 411.5 GiB，因此资源清理整体仍只能标为部分通过。

证据：

- `route-single-122b-start.json`
- `route-single-122b-quorum-1.json`
- `mixed-multi-routing.json`
- `run-mixed-multi-routing.py`
- `final-cleanup-summary.json`

## 7. 性能汇总

以下均为本轮小样本实测，不应直接视为稳定 SLA。

| 拓扑 / 模型 | 短 / 中 / 长解码速度 | 短 / 中 / 长 TTFT 均值 | 结果 |
|---|---|---|---|
| 单机 4B（Mac A） | 176.1 / 169.8 / 167.4 tok/s | 0.216 / 0.229 / 0.255 s | 稳定 |
| 单机 4B（Mac B，短） | 169.4 tok/s | 0.189 s | 稳定 |
| 单机 122B（Mac A） | 60.3 / 59.6 / 59.4 tok/s | 0.451 / 0.418 / 0.481 s | 稳定 |
| 多机 4B | 127.4 / 132.6 / 131.6 tok/s | 0.187 / 0.190 / 0.234 s | 稳定，但比单机 4B 低约 24% |
| 多机 122B | 59.7 / 57.8 / 56.4 tok/s | 0.207 / 0.257 / 0.423 s | 接近单机 |
| 多机 GLM | 24.3 / 24.1 / 23.7 tok/s | 0.341 / 0.485 / 0.744 s | 仅崩溃前数据，不可作为发布性能 |

性能证据为各模型的 `*-direct-short.json`、`*-direct-medium.json`、
`*-direct-long.json` 和单机 4B 的 `*-benchmark.json` 文件。

## 8. 长上下文与内容正确性

| 拓扑 / 模型 | Prompt tokens | HTTP / 保留 marker | 严格匹配 | 判定 |
|---|---:|---|---|---|
| 单机 4B，Mac A | 11,299 | 200 / 全部 | 是 | 通过 |
| 单机 4B，Mac A | 42,099 | 200 / 全部 | 是 | 通过 |
| 单机 4B，Mac B | 90,075 | 200 / 全部 | 否，增加标签 | 部分通过 |
| 多机 4B | 45,071 | 200 / 全部 | 否，增加标签 | 部分通过 |
| 多机 122B | 18,071 | 200 / 全部 | 是 | 通过 |
| 多机 GLM | 8,062 | 200 / 全部 | 是 | 该请求通过，但实例随后崩溃 |
| Auto Router | 42,099 | 409 `context_capacity` | 否 | 路由能力画像失败 |

模型切换本身不能掩盖内容错误。本轮 4B、122B 和 GLM 的复杂数学输出均不等于
`36419753088`；JSON factorization 也都没有满足 prime factors 语义。后续回归
必须同时验证 route provenance 和内容 oracle。

## 9. 资源清理与内存压力

### 9.1 普通实例

- 单机 4B/122B 停止后，Mac A 无 live instance、无 orphan，in-use 回到约
  6.5 GiB。
- 首轮多机 4B 停止后，A/B 均无 live instance、无 orphan。
- 独立多机 122B 最终两端停止。
- 卡住的第二个分布式 122B 被显式停止；原分布式 4B 仍保持 2/2 ready。
- 混合拓扑测试结束后，分布式 4B 与单机 122B 均通过各自的精确实例端点停止；
  A/B 最终活动实例、reservation、port reservation、orphan 和模型进程均为 0。

证据：

- `single-a-after-stop.json`
- `single-b-after-stop.json`
- `multi-4b-after-stop-a.json`
- `multi-4b-after-stop-b.json`
- `multi-122b-stop.json`
- `route-multi-122b-loading-stop.json`
- `route-multi-4b-after-sibling-stop-quorum.json`
- `final-cleanup-summary.json`

### 9.2 GLM wired 残留

GLM 前后无 live role 时的关键数据：

| 节点 | GLM 前 wired / in-use / pressure | GLM 后 wired / in-use / pressure |
|---|---|---|
| Mac A | 4.9 / 6.8 GiB / 0.99 | 190.7 / 192.4 GiB / 0.62 |
| Mac B | 226.7 / 235.0 GiB / 0.55 | 约 412.4 / 419.6 GiB / 0.19 |

两端本轮都增加约 185.8 GiB wired，接近单个 GLM shard。Mac B 在本轮前已经
有约 226.7 GiB wired，说明更早的运行也可能留下残留。

随后尝试双分布式 4B+122B 时，Mac B 曾达到：

- wired 约 443.3 GiB；
- in-use 约 455.2 GiB；
- `pressure_available_ratio: 0.13`。

同时，Resource Ledger 在 GLM 停止后仍报告 reservation 为 0、可用约
460.8 GiB。也就是说，准入计算没有扣除真实 wired/in-use，允许了一个物理上
风险极高的第二分布式实例。

证据：

- `pre-glm-status-a.json`
- `pre-glm-status-b.json`
- `post-glm-status-a.json`
- `post-glm-status-b.json`
- `post-glm-memory-later-a.json`
- `post-glm-memory-later-b.json`
- `route-ready-122b-status-b.json`
- `route-wait-122b-status-b.json`

### 9.3 本轮采用的安全边界

在 Mac B 处于上述状态时，停止继续增加纯双分布式压力是正确选择。后续重新测试前
至少满足：

- 两端无 live role、无 orphan。
- `pressure_available_ratio >= 0.35`，建议达到 0.50。
- 加载前 `in_use_ratio <= 0.66`。
- 加载前 `wired_ratio <= 0.60`。
- 计入约 43.2 GiB/节点的 4B+122B reservation 后，预计
  `in_use_ratio <= 0.75`。
- 加载后 pressure 低于 0.25、in-use 超过 75%、swap 持续增长，或相对基线
  新增超过约 64 GiB 非回收内存时立即停止。

在当前 GLM 后的 Mac B 状态下，安全压力边界为“不继续双模型压力测试”，而不是
降低并发后继续冒险。

## 10. 缺陷清单

### 10.1 P0

| ID | 缺陷 | 影响 | 复现证据 |
|---|---|---|---|
| HW-P0-01 | GLM 重复请求后 JACCL `Send failed -12`，协调进程 `-6` | 多机推理崩溃、恢复 502、quorum 丢失 | `multi-glm-complex.json`, `multi-glm-stop-after-cancel-failure.json` |
| HW-P0-02 | GLM 停止后每节点约 185.8 GiB wired 未释放；Ledger 仍按零 reservation 放行 | 后续加载可把机器推至 0.13 pressure，存在系统失稳风险 | `post-glm-memory-later-a.json`, `post-glm-memory-later-b.json`, `route-ready-122b-status-b.json` |
| HW-P0-03 | 已有分布式 4B 时，第二个分布式 122B 在 148/148 后无法完成同步 | 纯双分布式多模型常驻与路由不可用 | `route-multi-122b-quorum-1.json` 至 `route-multi-122b-quorum-5.json` |
| HW-P0-04 | 多机 4B 第二次加载时 worker `_share_object` 触发 `UnicodeDecodeError` | 停止/切换/重载不可靠，只能首次启动 | `multi-4b-reload-quorum-ready.json`, `multi-4b-reload-chat-ready.json` |
| HW-P0-05 | Mac B 的 legacy 用户 Agent 持续抢占 9100；bind 失败时会向默认 8000 发送 stop | 可误杀另一个 Agent 正在管理的有效模型实例 | `mac-b-duplicate-launchagent.txt`, `mac-b-after-port8000-failure.json` |

### 10.2 P1

| ID | 缺陷 | 影响 | 复现证据 |
|---|---|---|---|
| HW-P1-01 | `max_completion_tokens` 未生效 | 客户端 token/cost/latency 上限失真 | `single-routing-functional.json` |
| HW-P1-02 | 请求 logprobs 时仍返回 `null` | OpenAI 兼容 API 合约不完整 | `single-routing-functional.json` |
| HW-P1-03 | 强制工具调用仍无 `tool_calls` | Advertised tools capability 与实际不符 | `single-4b-tools.json`, `single-122b-tools.json`, `multi-4b-complex.json`, `multi-122b-complex.json` |
| HW-P1-04 | Router 拒绝已被直连 4B 处理的 42,099-token 输入 | Auto 模式错误拒绝可执行请求 | `single-decision-context-42099.json`, `single-4b-needle-3000.json` |
| HW-P1-05 | 混合拓扑结束时 4B 瞬时停在 `prefill_pending` 并从 routes 消失 | 健康采样/路由表有短暂错误摘除风险 | `mixed-multi-routing.json` |
| HW-P1-07 | 多个模型在严格数学、JSON prime factors 和工具调用任务上失败 | “请求成功”不能代表可用结果，路由质量评分缺少真实反馈 | 各 `*-math*.json`, `*-json.json`, `*-complex.json` |

## 11. 后续真实硬件回归门槛

修复后必须从干净机器状态重新执行，混合拓扑成功不能豁免纯双分布式验收。

### 11.1 环境门槛

- 两端只存在一个受支持的 Tokenity LaunchAgent。
- Agent、代码、MLX、mlx-lm 和模型 revision 完全一致。
- RDMA active，预检无 issue。
- 无 live role、无 orphan、无遗留端口 reservation。
- 满足第 9.3 节的内存安全边界；必要时重启两台机器后记录新基线。

### 11.2 单机推理门槛

- Mac A 的 4B/122B、Mac B 的 4B 各连续至少 30 请求，HTTP 成功率 100%，
  无生成线程死亡。
- 数学答案严格等于 `36419753088`。
- JSON 可解析且 prime factors 严格等于 `[3, 3, 5, 1783]`。
- 工具调用返回结构化 `tool_calls`。
- 4B 的 42K 与 90K marker 测试都严格匹配。
- stop 后 120 秒内无 live role/orphan，内存回到基线 +16 GiB 以内。

### 11.3 单机路由门槛

- 20 次显式交替和 20 次 Auto 交替保持 100% 正确 model/instance/revision。
- sticky、topic change、model lock、allowed IDs、unsupported modality 全部符合
  预期。
- `max_completion_tokens`、logprobs、tools 和 context capacity 合约全部修复。
- 并发 2/4 全部完成；无 5xx；queue wait p95 小于 20 秒。
- SSE 只有一个 `[DONE]`、无 malformed；取消后 10 秒内释放并可恢复。

### 11.4 独立多机模型门槛

每个模型都需：

- 连续执行至少 3 个 start → ready → infer → cancel → recover → stop → reload
  周期。
- 每次达到 2/2 quorum，revision 和 rank mapping 正确。
- 4B/122B 各至少 20 个连续复杂请求，GLM 至少 20 个，并且零 5xx、零 JACCL
  异常、零 generation thread death。
- 长上下文严格匹配，不只检查 marker 存在。
- stop 后两端 wired/in-use 在 120 秒内回到各自测试前基线 +16 GiB 以内，
  pressure 不低于 0.35。

GLM 在上述稳定性和清理门槛全部通过前不得加入 Auto Router 候选池。

### 11.5 纯双分布式多模型路由门槛

- 分布式 4B 和分布式 122B 同时在 180 秒内达到 2/2 ready。
- 两个 route 持续可见，至少 20 次 Fast/Quality 交替全部选中预期实例。
- 显式模型、sticky、topic change、lock、allowed IDs 全部通过。
- 并发 2/4、SSE、取消恢复全部通过。
- 精确停止 4B 后 122B 保持 2/2 且继续服务；重载 4B 后重新进入候选池。
- 反向停止/重载 122B 再执行一次。
- 两个实例至少重复 3 个共同启停周期，不得再次出现 148/148 同步卡住或
  `_share_object` 解码异常。
- 最终停止全部实例并记录 A/B 清理证据。

## 12. 证据完整性说明

- 本报告只依据 `dist/validation-2026-07-26/` 中的真实硬件产物。
- 基准结果的样本量较小，适合发现数量级回退，不足以建立长期 SLA。
- HTTP 200、quorum ready 和语义正确性分别判定，没有用其中一项替代另外两项。
- 纯双分布式路由失败后采用混合拓扑，是基于实际内存压力的安全措施；本报告没有
  将该替代结果包装成原目标通过。
- 混合拓扑最终停止与账本/进程清理已有 `final-cleanup-summary.json` 证据；该证据
  同时确认 wired 内存没有回到健康基线，因此不能视为完整内存清理通过。

## 13. 首轮之后的修复与回归状态

以下修复是在本报告第 1–12 节证据冻结后实现的，不能反向改变首轮判定：

- Agent shutdown 只停止本 Agent 实际拥有且仍运行的 coordinator；空 Agent、
  bind 失败、已退出实例和 worker rank 不再探测或停止默认 8000。
- 完整安装器和 code-update 安装器都会精确 bootout 并移除 legacy
  `local.tokenity.node-agent` 用户服务。
- 新模型准入同时受 reservation、实时 wired/in-use 和
  `memory_pressure` 限制，并保留 25% 最小内存余量。
- GLM/JACCL 流取消后 runtime 立即 fail-closed，后续请求返回 503 并要求
  双 rank 重载，不再复用可能失步的 collective。
- JACCL 控制帧新增 magic/version、operation epoch、sequence、长度上限和
  CRC32；验证失败时结构化 fail-fast，不再把未验证字节交给 `pickle.loads`。
- 停止后的 HTTP/collective 端口隔离 30 秒，立即重载会自动选取新端口。
- `max_completion_tokens`、logprobs 输出、工具状态格式化、Qwen 嵌套
  `text_config.max_position_embeddings` 和 stopped 实例陈旧状态已修复。

在隔离的 9200 验证 Agent 上进行的真实硬件回归确认：

- Mac B 的实时可准入内存为 0；4B 加载在创建进程前以
  `resource_admission` 409 拒绝，且没有 reservation 或 role 泄漏。
- Mac A 单机 4B 的 `max_completion_tokens: 3` 严格返回 3 tokens。
- 42,099-token Auto 决策不再触发 `context_capacity`，4B 候选为 eligible。
- 停止后的 `actual_memory_bytes` 为 null、`health_ready` 为 false。
- 使用原首选端口立即重载时自动从 8100/30100 切换到 8101/30101，
  新实例达到 1/1 并返回 `RELOAD_OK`。
- logprobs 返回真实 top token，包括候选 `Okay: -6.375`，不再为 null。
- A/B legacy 用户 Agent 已卸载，plist 已移动到可恢复备份目录；系统 9100
  Agent 保持在线。

由于 Mac B 最终仍有约 412 GiB wired、实时可准入内存为 0，本轮没有绕过新
准入规则重跑双机 4B reload、GLM cancel/reload 或纯双分布式路由。相关修复已经
通过自动化测试，但仍必须在两台机器恢复干净内存基线后完成第 11 节真实硬件门槛。

回归证据：`post-fix-regression-summary.json`。

## 14. 最终工程与安装包验证

修复后重新执行 `scripts/verify-stable-baseline.sh`，结果如下：

| 验证项 | 结果 |
|---|---|
| Python 测试 | 167 passed，8 skipped |
| Swift 测试 | 114 passed，0 failed |
| Chat / Sidebar / Models 明暗主题视觉冒烟 | 5 passed |
| 首次安装引导 | 5 passed，包含首次展示、真实 sheet 关闭/重开、完成状态持久化和明暗主题 |
| Runtime Bootstrap | 4 passed，包含固定 catalog、缓存发现和篡改拒绝 |
| Release App 构建与严格代码签名结构校验 | 通过 |
| Installer shell 语法、`git diff --check` | 通过 |

重新生成并校验的产物：

| 产物 | 大小 | SHA-256 | 校验 |
|---|---:|---|---|
| `Tokenity-0.1.0.dmg` | 192,868,704 bytes | `bb8fd9713343d60ecf8748c528084dfa7507d2d2dc37805d6f19fb829d75b7e1` | SHA、`hdiutil verify`、挂载布局通过 |
| `Tokenity-NodeAgent-Runtime-2026.07.26.1-macos-arm64.pkg` | 188,069,414 bytes | `e1b834df9916d6b12bbaff2364b7591432e880f6668e141d4465fee5cbe95f1b` | catalog/大小/SHA/Runtime manifest 通过 |
| `Tokenity-NodeAgent-0.1.0.pkg` | 102,375 bytes | `d23e8530606f2d86eec613e6a2987d6f5a775be4cc59f7987ecfd3c9f7023f67` | payload 与当前 Agent/server 源码逐字节一致 |

DMG 挂载后确认包含：

- `TokenityControl.app`
- `Applications -> /Applications`
- `Install Tokenity Node Agent.pkg`
- `Runtime Catalog.json`
- `Runtime Installer.sha256`
- `README.txt`

DMG 中可见的 Runtime 安装器是 App 内同一份固定 PKG 的相对符号链接；两者
逐字节一致。Runtime 为 arm64、Python 3.12.13、MLX 0.32.0、
MLX-LM 0.31.3，payload tree SHA-256 为
`886a97ee49bf17895c01e4918cf46855a827889227fae86e5b064180e03f2411`。

发布边界：

- 此 DMG 已适合内部实机安装验证，但 App 仍是 ad-hoc 签名，两个 PKG 未使用
  Developer ID Installer 签名，尚未公证；Gatekeeper 的公开发行验收会拒绝。
- Runtime 中 MLX/JACCL 二进制的最低系统版本是 macOS 26.2。
- 默认 DMG 包含 App、Agent 和 Runtime，不包含数百 GB 的模型权重。
- 完整 DMG 面向首次安装；较小的 Agent update PKG 只用于已安装相同 Runtime
  和 system LaunchDaemon 的机器，不能替代首次安装。
- 产物基于当前工作树字节生成。当前 HEAD 为
  `3b7335bcf1ec86b55cb93dc3535ddfb65696e227`，但工作树包含本轮尚未提交的修复，
  因此公开发布前应先形成可追溯 commit，再从该 commit 重建签名产物。
- 本轮没有把更新 PKG 安装到生产 9100 Agent；该动作需要管理员批准并会重启
  系统服务。生产 Agent 保持原版本，隔离验证 Agent 已停止。

最终安装包验证证据：`package-validation-summary.json`。

## 15. 重启后稳定性修复复验 — 2026-07-27

本节覆盖第 13 节因 Mac B wired 残留而未能执行的真实硬件门槛。复验只使用
隔离的 9200 Agent；本节复验期间生产 9100 Agent 保持原 revision，未安装、
修改或重启。复验完成后的授权生产部署见第 15.7 节。
两端计入结果的模型子进程均明确继承各自的
`TokenityCode-Codex-Validation` 目录作为 `PYTHONPATH` 和
`TOKENITY_CODE_ROOT`。验证 revision 为
`d66e134f1806a9ed7572322985fd7b3b07735bf54f02d66487175b66323a9020`。

### 15.1 新增修复

- GLM/JACCL 启动时将 MLX wired residency limit 设为 0；模型仍完整
  materialize，但不再把约 198 GiB shard 固定为 macOS wired 内存。
- JACCL rank 使用带 magic、version、operation epoch、sequence 和 CRC32 的
 停止帧共同退出 generation loop；协调端通过管理端点启动自然 Uvicorn
  shutdown，不再立即用 SIGTERM 把正常退出改成 `-15`。
- GLM 顺序生成的客户端取消增加 10-word CPU collective，把 rank 0 的
  cancel 状态同步给所有 rank。取消后的 runtime 立即 fail-closed，后续请求
  返回 503；停止和重载不再遗留卡住的 worker。
- 同一停止哨兵扩展到全部分布式 JACCL 模型，消除 Qwen 重载时协调端先离开
  `_share_object`、worker 随后只能被 SIGKILL 的竞态。
- 回归测试均先复现失败再修改实现；相关生命周期测试位于
  `tests/test_distributed_openai.py` 和 `tests/test_node_agent.py`。

### 15.2 GLM/JACCL

**状态：稳定性与清理通过；语义质量仍不纳入 Auto。**

- 三个取消/恢复导向的完整周期均达到 2/2 Ready；其中一次真实客户端在首个 SSE token
  后发送 TCP RST，runtime 立即进入 `failed / stream_cancelled`，后续请求
  返回 503。
- 另一个严格长上下文完整周期达到 2/2 Ready；7,606-token 输入精确返回三枚
  分散 marker，协调端和 worker 停止退出码均为 0。因此本轮共有四个有效
  GLM 完整启停周期。
- 精确停止后协调端和 worker 均为退出码 0；取消后使用新 instance 完整重载，
  随后 10/10 连续请求成功。
- 两个非取消轮合计 20/20 请求为 HTTP 200 且严格 marker 正确。
- 约 198 GiB/节点权重 materialize 时，wired 仍约 5–8 GiB；最终停止后的
  0/30/60/90/120 秒采样均为 0 活 rank、0 reservation、0 orphan。
- 120 秒时 Mac A 为约 4.86 GiB wired / 6.35 GiB in-use / 0.99 pressure；
  Mac B 为约 7.48 GiB wired / 14.31 GiB in-use / 0.98 pressure。

早期若干 GLM 试验实际子进程仍来自 `${TOKENITY_CODE_ROOT}`，没有加载本轮
修复。它们保留作诊断证据，但不计入上述通过/失败统计。

### 15.3 独立分布式 Qwen

| 模型 | 最终实现完整周期 | 连续请求 | 取消恢复 | rank 停止 | 严格语义 |
|---|---:|---:|---|---|---|
| Qwen3.5 4B | 3 | 20/20 HTTP 200 | 通过 | 每轮 0/0 | 0/20 |
| Qwen3.5 122B | 3 | 20/20 HTTP 200 | 通过 | 每轮 0/0 | 0/20 |

补充严格长上下文复验：

| 模型 | Prompt tokens | HTTP | 三枚 marker 严格匹配 | 停止 |
|---|---:|---:|---|---|
| Qwen3.5 4B | 49,104 | 200 | 是 | 0/0 |
| Qwen3.5 4B | 105,104 | 200 | 是 | 0/0 |
| Qwen3.5 122B | 27,362 | 200 | 是 | 0/0 |
| GLM-5.2 | 7,606 | 200 | 是 | 0/0 |

机器可读证据为 `strict-long-context-results.json`；四次响应内容均严格等于
`ALPHA-7F3C91|BETA-2D8A44|GAMMA-9E1B62`，没有附加标签或解释。

严格语义仍失败，不能用稳定性通过掩盖：

- 4B 对数学题稳定返回错误的 `3655399992`，JSON 稳定返回错误因数
  `[5, 13, 1247]`。
- 122B 对数学题稳定返回错误的 `36543209990`，JSON 稳定返回错误因数
  `[3, 5, 13, 41, 127]`。
- 因此 `HW-P1-07` 仍为失败；本节只关闭传输、collective、取消和重载稳定性。

### 15.4 4B + 122B 纯双分布式多模型

**状态：通过。**

- 连续三个共同周期均先让双机 4B 达到 2/2，再在其常驻时加载双机 122B。
  两个实例均在约 15 秒内达到 2/2；没有再出现 148/148 后同步卡住或
  `_share_object` 解码异常。
- 显式模型交替 20/20、Fast/Quality Auto 交替 20/20，instance 和 model
  revision provenance 均匹配。
- 补充的 Balanced 纯双分布式轮次中，10/10 router decision 和 4/4 真实网关
  推理均选择预期 4B；响应的 model、instance、revision 路由头全部匹配，
  四次内容严格等于各自的 `BALANCED_OK_n`。
- sticky、topic change、model lock、allowed instance IDs 均选择预期实例。
- 并发 4/4 完成；Auto SSE 恰有一个 `[DONE]` 且无 malformed event。
- 精确停止 4B 后 122B 保持 2/2 并继续推理，重载 4B 后重新进入候选池；
  反向停止/重载 122B 也通过。
- 三个共同周期最终四个 rank 均为退出码 0。最终 120 秒资源观测为 0 活 rank、
  0 reservation、0 orphan，pressure 为 0.98–0.99。

### 15.5 OpenAI 合约与 UI

- 真实分布式 4B 的强制 `get_weather` 请求返回结构化 `tool_calls`，
  `finish_reason: tool_calls`，参数为 `{"city":"Shanghai"}`。
- 同一真实模型请求同时设置 `max_tokens: 100` 和
  `max_completion_tokens: 3` 时严格生成 3 tokens；`logprobs` 非 null 且含
  3 个 content entry。
- `scripts/verify-stable-baseline.sh`：Python 176 passed / 8 skipped，
  Swift 114 passed / 0 failed，App bundle 构建通过。
- 使用 `scripts/run-tokenity-control-app.sh` 真实启动 UI 后，Overview 显示
  Cluster Stopped、0 resident models；Mango/Kiwi 均 online、Thunderbolt
  Ready，内存和 pressure 与 9200 Agent 实况一致。Cluster 与 Models 页面
  均可正常打开。

完整机器可读摘要：

`dist/validation-2026-07-26/stability-fix-after-reboot-20260726-2233/stability-fix-validation-summary.json`

### 15.6 最终重建产物

本节代码和文档更新后重新执行 `scripts/package-tokenity-dmg.sh`，并独立执行
checksum、`hdiutil verify`、只读挂载布局和 Runtime catalog 校验：

| 产物 | 大小 | SHA-256 | 校验 |
|---|---:|---|---|
| `Tokenity-0.1.0.dmg` | 192,921,337 bytes | `a864a739ab8e939e9ba43d21eba98d67dd70daefcf6e8d50fe7033f2e35b1b45` | checksum 与 `hdiutil verify` 通过 |
| `Tokenity-NodeAgent-Runtime-2026.07.26.1-macos-arm64.pkg` | 188,079,649 bytes | `a1583f6028277b987125a726068e8fe048bcb1e80ae88fb228fe488d621b807f` | checksum 与 Runtime catalog 通过 |

只读挂载确认包含 App、Applications 链接、内置 Node Agent PKG、Runtime
Catalog、Runtime checksum 和 README。Runtime PKG 仍未签名；本产物适合内部
实机验证，不满足公开发行的 Developer ID Installer/公证门槛。

### 15.7 生产 9100 升级 — 2026-07-27

用户明确授权后，将已通过上述实机验证的 revision
`d66e134f1806a9ed7572322985fd7b3b07735bf54f02d66487175b66323a9020`
部署到两端生产 9100 Agent。

- 部署前两端均为 0 活动 role、0 orphan、0 reservation；已安装 Runtime 的
  MLX 0.32.0 / mlx-lm 0.31.3 与更新包要求一致。
- 发现原 `Tokenity-NodeAgent-0.1.0.pkg` 早于最终修复代码生成，未直接安装。
  从已验证工作树重新构建 code-only 更新包，SHA-256 为
  `6dee11f72cfc13f50a88ea83d9cedfa9526f57dd699ef0214b785b49f76eb815`，
  内置期望 revision 与实机验证 revision 一致。
- 按 Mac A、Mac B 顺序升级。每端安装前均保存
  `${TOKENITY_CODE_ROOT}/tokenity` 备份；安装器返回成功并写入
  `ai.tokenity.node-agent.update 0.1.0` receipt。
- 升级后两端 9100 均在线并发布 `d66e…9020`；RDMA `rdma_en4` /
  `rdma_en5` active，无 error。两端均识别 3 个模型，且仍为 0 活动实例、
  0 reservation、0 port、0 orphan；pressure 分别为 0.99 / 0.98。
- DMG 内 App 已安装至 `/Applications/TokenityControl.app`，bundle ID
  `ai.tokenity.control`，版本 0.1.0；代码签名结构验证通过并已启动。
- 此次部署验收没有启动模型；模型生命周期和双机推理证据沿用第 15.2–15.5 节
  对相同 revision 的隔离 9200 实机回归。

部署机器可读证据：

`dist/validation-2026-07-26/stability-fix-after-reboot-20260726-2233/production-upgrade-2026-07-27.json`

### 15.8 GLM→Qwen 122B 生产稳定性修复 — 2026-07-27

**状态：针对用户复现的 GLM 常驻后加载 Qwen 122B、停止再热重载路径，修复并在
生产 9100 双机环境通过。**

用户在真实 UI 中先加载 GLM-5.2，再加载 Qwen3.5-122B-A10B 时，Qwen 显示
`Failed`。本轮从 UI 到分布式 runtime 共定位并关闭三个相互叠加的问题：

- UI 的后台节点快照可能暂时不含正在启动的新实例，旧逻辑会在这一个快照内把
  GLM 恢复成当前 active model，导致 Qwen 启动探针被错误发给 GLM。现在只要
  `activeModelLoadID` 存在，就保留正在加载的实例身份，不再被瞬时旧快照覆盖。
- 第二个 resident communicator 存在时，direct JACCL 可能在 `recv` 返回
  `-12`。现在全部 direct JACCL 请求都统一升级为 `jaccl-ring`，不再只对 GLM
  特判。
- Qwen 权重命中缓存后可在健康监控切换生成模式前完成加载，worker 已进入
  request-control loop，而 rank 0 仍在模型 collective，形成
  `protocol mismatch`。现在在 `ResponseGenerator` 启动前同步固定 JACCL
  模型使用 sequential engine、关闭 prompt cache，并在 `load_default()`
  返回时立即把 provider 标为非 batchable，从初始化顺序上消除该竞态。

最终 code-only Agent revision 为
`1ac633861c314fe36c0c3be560d63abeab1975d51df8f970028cb58b81523178`，
已经安装到 Mac A `<node-a-lan-ip>:9100` 和 Mac B `<node-b-lan-ip>:9100`。
两机清洁重启后均发布相同 revision，RDMA `rdma_en4` / `rdma_en5` active，
基线为 0 instance、0 role、0 reservation。

真实 UI 生产回归：

- 先加载 GLM，实例
  `E8607F34-EFCC-4FC0-B5D4-956D66F20742`，服务端口 8000、
  collective 起始端口 30020，最终 2/2 Ready。
- 在 GLM 常驻时首次加载 Qwen 122B，实例
  `A71AEF8A-F72E-4799-BCC5-C58BB4780170`，UI 与网关均达到
  Ready / Loaded。
- 通过 UI 精确停止 Qwen 后，reservation 从 2 降至 1，GLM 未受影响。
- 随即通过 UI 热重载 Qwen，得到新实例
  `AD5EA286-286B-4FDE-A54F-8DD469B40828`，服务端口 8002、
  collective 起始端口 30024；旧端口处于 quarantine，未被错误复用。缓存命中
  的快速加载仍达到 2/2 Ready，UI 显示 Loaded。

热重载后 Qwen 的 2 次非流式与 2 次流式请求全部 HTTP 200，分别严格返回
`HOT-1`、`HOT-2`、`HOT-3`、`HOT-4`；GLM 的 2 次复验也全部 HTTP 200 且
严格返回预期 marker。此前同一修复族还完成 Qwen 4 次非流式、3 次流式、
3 轮 GLM/Qwen 交替请求及 12 秒空闲后的 Qwen 请求，均为 HTTP 200。

最终两实例在两端均保持 2/2 rank quorum、`health_ready=true`、
`status_stale=false`，无 orphan。对两实例、两台主机的当前日志扫描
`Recv failed`、`Send failed`、`protocol mismatch`、`Traceback`、
`Exception in thread`、`error code -12`，匹配数为 0。
2026-07-28 00:00:11 +08:00 的最后一次只读回查仍为两个 coordinator
进程、两个 worker 进程运行，Mac A / B 的 pressure available ratio 分别为
0.98 / 0.91。

代码回归为 Python **178 passed / 8 skipped**、Swift **115 passed /
0 failed**；新增测试覆盖 UI 瞬时旧快照、全部模型 direct JACCL ring fallback、
Qwen sequential engine，以及热重载生成线程启动前的同步模式准备。相关新增
测试在修复前均可复现失败。

最终产物：

| 产物 | 大小 | SHA-256 |
|---|---:|---|
| `Tokenity-0.1.0.dmg` | 192,890,666 bytes | `8c369c28238f74cc830d2212ea25dfd03b012ff4561cf921c3664657bc15cd41` |
| `Tokenity-NodeAgent-0.1.0.pkg` | 105,181 bytes | `5777eae630cdf3e25dd3ee66a9c570af3af6c3695595efdffed8c7695aa413be` |
| `Tokenity-NodeAgent-Runtime-2026.07.26.1-macos-arm64.pkg` | 188,082,165 bytes | `ea545ac405d587c562442ccf0f48572ba3a375ae1e40f715de83226f9ae50b96` |

DMG 中的 release App 已安装至 `/Applications/TokenityControl.app`，bundle ID
为 `ai.tokenity.control`、版本 0.1.0，代码签名结构校验通过；安装前后可执行文件
SHA-256 均为
`d94632107fed18cf8992d9ed10500eb1050e2ee8becb53e7320cab67f855d53f`。
为避免中断当前实例，没有终止正在操作和验证的 debug App。交接时 GLM 与热重载后的
Qwen 仍在生产 9100 常驻运行。

本轮机器可读证据：

`dist/validation-2026-07-26/stability-fix-after-reboot-20260726-2233/glm-qwen122-production-fix-2026-07-27.json`
