# Tokenity 后续开发先验知识与操作基线

> 最后整理：2026-07-23
> 适用范围：Tokenity Control、Node Agent、MLX/MLX-LM 推理服务、多机与多实例开发
> 原则：本文记录开发约束和已验证经验。环境状态可能变化，开始工作时仍需重新检查。

## 1. 唯一事实来源

Tokenity 唯一正确的源码仓库是：

```text
/path/to/Tokenity-Stable
```

所有源码读取、修改、测试、构建、打包和 UI 启动都必须从这个仓库进行。

以下目录是旧工作区，只允许作为只读参考：

```text
/path/to/MLX-Distributed
```

特别禁止从以下旧路径修改、构建或启动 Tokenity UI：

```text
/path/to/MLX-Distributed/apps/TokenityControl
```

`scripts/run-tokenity-control-app.sh` 中如果出现旧 UI 的绝对路径，它只用于精确清理旧原型进程，不代表正确源码或启动位置。

开始任何工作前先执行：

```bash
cd /path/to/Tokenity-Stable
pwd
git status -sb
```

不得使用 `git reset --hard`、`git checkout --`、`git clean` 等可能丢失现有改动的命令。

## 2. 正确路径

| 用途 | 路径 |
| --- | --- |
| 唯一源码仓库 | `/path/to/Tokenity-Stable` |
| SwiftUI 工程 | `/path/to/Tokenity-Stable/apps/TokenityControl` |
| UI 启动脚本 | `/path/to/Tokenity-Stable/scripts/run-tokenity-control-app.sh` |
| UI 构建脚本 | `/path/to/Tokenity-Stable/scripts/build-tokenity-control-app.sh` |
| 完整基线验证 | `/path/to/Tokenity-Stable/scripts/verify-stable-baseline.sh` |
| 默认 Debug App | `/path/to/Tokenity-Stable/apps/TokenityControl/.build/arm64-apple-macosx/debug/TokenityControl-Stable.app` |
| App 可执行文件 | `TokenityControl-Stable.app/Contents/MacOS/TokenityControl` |
| Stable Bundle ID | `ai.tokenity.control.stable` |
| Runtime Python | `${TOKENITY_RUNTIME_PYTHON}` |
| 模型根目录 | `${TOKENITY_MODEL_ROOT}` |
| 正式安装的共享代码 | `${TOKENITY_CODE_ROOT}` |

正确启动 UI：

```bash
cd /path/to/Tokenity-Stable
./scripts/run-tokenity-control-app.sh
```

不要通过名称相似的旧 `.app`、旧 Bundle ID 或旧工作区启动。当前稳定开发包名称应为：

```text
TokenityControl-Stable.app
```

构建产物路径可能随 Swift 工具链变化；需要精确定位时，以构建脚本输出或 `swift build --show-bin-path` 为准。

## 3. SwiftUI 代码分工

| 文件 | 职责 |
| --- | --- |
| `TokenityControlApp.swift` | App 生命周期、窗口、菜单栏入口和共享 Store 注入 |
| `TokenityStore.swift` | 节点轮询、集群、模型生命周期、Chat、实例和错误状态的唯一业务状态源 |
| `Models.swift` | HTTP DTO、节点、模型、Agent contract、runtime 和 UI 状态模型 |
| `Views.swift` | 主窗口导航、Overview、Nodes、Models 等通用页面 |
| `ChatViews.swift` | Chat 工作区、会话、输入框、消息与指标 UI |
| `ChatStreamParsing.swift` | SSE、thinking 标签、reasoning/content 增量流解析 |
| `MarkdownRendering.swift` | 流式 Markdown、代码块、表格和列表渲染 |
| `MenuBar.swift` | 状态栏图标、菜单内容、健康级别和节点快照 |

主窗口、Models、Chat、Overview 和 MenuBar 必须使用同一个由 AppDelegate 持有的 `TokenityStore`。

禁止：

- 在不同页面分别创建 Store。
- 在 MenuBar 中重新推导一套与主窗口不同的模型状态。
- 用本地 UI 文案或动画覆盖后端真实失败。
- 把“进程存在”“模型目录存在”或“`/v1/models` 有名称”当成 Loaded。

## 4. Python 后端代码分工

| 文件 | 职责 |
| --- | --- |
| `tokenity/node_agent/agent.py` | Node Agent HTTP API、能力握手、本机 rank/实例控制、gateway |
| `tokenity/serving/distributed_openai.py` | MLX/MLX-LM 推理、warmup、SSE 和 OpenAI API |
| `tokenity/serving/readiness.py` | readiness、生命周期和推理证据 |
| `tokenity/process/supervisor.py` | 按 `(role, instance_id)` 隔离的本机进程监督 |
| `tokenity/control/instances.py` | 多实例、端口、资源准入、request lease 和生命周期 |
| `tokenity/mlx/launcher.py` | MLX rank 启动和分布式环境 |
| `tokenity/mlx/hostfile.py` | 数据面拓扑兼容；不得重新引入生产 SSH 编排 |

## 5. 控制面、数据面和端口

产品调用链必须保持：

```text
SwiftUI / TokenityStore
    -> Coordinator Node Agent typed HTTP :9100
    -> Worker Node Agent typed HTTP :9100
    -> 每台 Agent 只监督本机 rank / instance
    -> MLX Ring 或 JACCL/RDMA 负责 tensor 数据面
```

端口约定：

- Node Agent 控制面：`9100`
- 外部稳定 OpenAI 兼容入口：`http://<coordinator>:9100/v1`
- rank-0 runtime 私有端口：通常为 `8000`，也可能由实例动态分配
- 普通客户端应使用 `9100/v1`，不要把私有 runtime 端口作为默认入口

产品代码中禁止：

- SwiftUI App 执行 SSH。
- Coordinator 通过 SSH 启动 worker rank。
- launcher、supervisor、模型加载或多实例调度依赖 SSH。
- 使用 SSH 复制源码、同步模型或替代正式 Agent 部署。

SSH 只能用于开发者只读诊断。Node Agent 正式升级必须走 installer/launchd。

## 6. Agent 版本和能力握手

Control UI 源码、当前运行的 UI、远端 Node Agent 和远端 runtime 可能不是同一个版本。

每次诊断必须分别确认：

- 当前 UI 对应哪个仓库和 App bundle。
- `/v1/node/info` 返回的 Agent 版本、代码 revision 和 `agent_contract`。
- Agent 声明的 capabilities。
- runtime readiness 中的 instance、world size、rank 和 connection mode。

已建立的能力协议包括：

- `cluster_runtime`
- `instance_quorum`
- `instance_runtimes`
- `managed_instances`
- `native_mtp`

兼容原则：

1. 旧 Agent 没有 `agent_contract` 时，可以进入明确受限的 legacy 路径。
2. legacy Ready 必须由真实推理探针、健康服务和完整角色证据支持。
3. 现代 Agent 已声明某项能力却缺失对应数据时，必须显示 Warning。
4. 多节点只有部分 runtime metadata 时，必须 Warning，不能假装 Ready。
5. 不得把旧 Agent 的“字段不存在”与现代 Agent 的“字段异常缺失”混为一谈。

## 7. 模型 Loaded 和状态栏状态

模型进入 Loaded/Ready 至少需要：

- 模型实例身份明确。
- 后端服务处于真实 ready。
- rank quorum 与预期 world size 一致。
- rank 角色、instance ID 和 connection mode 匹配。
- 权重已物化，受控 warmup 已完成。
- 真实推理探针成功。
- 当前状态不是 stale heartbeat。

不能作为 Loaded 的充分证据：

- 模型目录存在。
- 模型出现在 inventory。
- `/v1/models` 返回名称。
- 某个进程仍存在。
- provider 对象已经创建。
- UI 仍保留上一次绿色状态。

状态栏图标映射本身已经验证正常：

- Ready：`checkmark.circle.fill`
- 正在生成且仍健康：`ellipsis.message.fill`
- Warning/Offline 不能被生成状态覆盖

如果模型可用但 Logo 错误，优先检查状态计算和协议兼容，不要先替换图标资源。

## 8. 节点在线状态

网络轮询可能出现瞬时失败。

当前规则：

- 一次 Node Agent 轮询失败不立即发布 Offline。
- 连续两次失败后才将节点标记为 Offline。
- 成功刷新后清零失败计数。
- UI 不应因一个短暂 polling miss 在全局状态栏闪烁 Offline。

诊断 Offline 时要区分：

- Agent 真正停止。
- 端口不可达。
- 短暂请求超时。
- UI 使用了旧节点地址。
- Agent 正在运行但返回旧协议。

## 9. 多实例隔离

所有角色、runtime、日志、停止操作和状态查询必须按 instance 隔离。

核心身份是：

```text
(role, instance_id)
```

不能只按全局 role 管理进程。

必须保持：

- `cluster_runtimes` 按 instance ID 存储。
- 旧 `cluster_runtime` 只作为单值兼容字段。
- 活跃模型状态优先读取完全匹配的 instance。
- 停止一个实例时，只清理该实例。
- 一个实例失败时，健康 sibling 继续运行和路由。
- 活跃实例退出后，可以恢复到仍 Loaded 的 sibling。
- request lease 未释放时不能误卸载正在服务的实例。

Coordinator 必须稳定保持 rank 0。节点移除、重新加入或排序变化时，不能仅根据数组顺序偷偷更换 coordinator。

## 10. Chat 无输出的排查链路

Chat 无输出不能只看 Chat 页面。必须沿完整链路检查：

```text
Chat UI
  -> TokenityStore 构造 stream=true 请求
  -> Coordinator :9100 gateway
  -> 目标 instance / rank-0 runtime
  -> MLX generation
  -> SSE response
  -> URLSession 增量读取
  -> SSE / thinking parser
  -> Store 发布增量 token
  -> SwiftUI 渲染
```

重点经验：

- 首次请求不能承担未完成的权重物化和 Metal 编译。
- warmup 必须在 Ready 前完成，并且可取消、有 deadline。
- 长 prefill 可以发送合法 SSE comment/keepalive，但不能伪装成内容 token。
- 非流式 fallback 只能用于明确的传输失败。
- malformed SSE、服务端错误和生成错误不得被自动 fallback 掩盖。
- reasoning 与 final content 必须分离；跨 chunk 的 `<think>` 标签不能泄漏到最终答案。
- 用户停止、卸载模型或停止集群后，必须拒绝迟到 token。
- 循环重复输出需要显式终止并给出原因。

## 11. 内存与状态指标

系统内存、进程内存和 MLX 内存不是同一个指标。

UI 需要区分：

- 系统总内存、使用量和可回收文件缓存。
- rank 进程 RSS。
- macOS `phys_footprint`。
- MLX active/peak/cache。
- 权重估算、KV/prompt 可用量。
- 指标采样时间和 stale 状态。

如果 Agent 不支持新的 runtime memory 字段，应显示未知或 legacy 状态，不能显示伪精确值。

## 12. 固定验证流程

每次开发完成后至少执行：

```bash
cd /path/to/Tokenity-Stable
git status -sb
git diff --check
./scripts/verify-stable-baseline.sh
```

随后启动正确 UI：

```bash
./scripts/run-tokenity-control-app.sh
```

人工 UI/硬件验证至少覆盖：

1. Nodes 页面中的节点恢复 Online。
2. 内存数据真实显示，缺失时不伪造。
3. 创建集群后 coordinator/rank 规划正确。
4. 模型从 Loading 进入真实 Loaded。
5. Chat 显示 Ready。
6. 发送短提示词能得到实际流式输出。
7. MenuBar 显示 Ready 图标。
8. 生成时图标正确切换。
9. 停止模型后所有相关 rank 退出。
10. 停止一个实例不会影响 sibling。

不能只凭单元测试声称真实硬件通过；真实硬件、真实 UI 和真实推理需要分别记录证据。

## 13. 当前已验证基线快照

截至 2026-07-23：

- Python：`81 passed, 8 skipped`
- Swift：`87/87 passed`
- `TokenityControl-Stable.app` 构建成功
- 双机 GLM 模型可以进入 Loaded/Ready
- Chat 可以生成并返回指定内容
- MenuBar 的 Ready 状态计算和图标映射测试通过

这些数字是历史快照，不是永久保证。后续修改必须重新运行基线。

## 14. Git 和发布注意事项

GitHub 仓库：

```text
HeyZhey/Tokenity
```

截至本文整理时：

```text
branch: feature/native-mtp-mvp
commit: 4594646a7d267a7c0e34e4c8d412d098540cea72
```

远端状态和分支可能变化，后续发布前必须重新检查。

当前本机存在两份内部文档，包含本机绝对路径、SSH 身份和内网拓扑，不应被无差别加入公开提交：

```text
docs/stability-audit-2026-07-17.md
docs/tokenity-stability-performance-multi-instance-session-prompt.md
```

因此发布前不要盲目执行 `git add -A`。应检查 `git status`，显式选择文件，或先对内部文档脱敏。

私钥、token、SSH 配置、模型权重、runtime 环境、`.venv`、Swift `.build`、日志、缓存和测试产物不得提交。

## 15. 后续任务的开工检查表

- [ ] 当前目录是 `/path/to/Tokenity-Stable`
- [ ] 没有从旧 `MLX-Distributed` UI 启动
- [ ] 已记录 `git status -sb`
- [ ] 已保护用户现有未提交修改
- [ ] 当前 UI bundle 是 `TokenityControl-Stable.app`
- [ ] 主窗口与 MenuBar 共用同一个 Store
- [ ] 已确认 Agent contract 和远端部署版本
- [ ] Loaded 来自真实 readiness 和推理证据
- [ ] 所有角色和 runtime 按 instance ID 隔离
- [ ] 外部客户端使用 coordinator `:9100/v1`
- [ ] 产品路径没有 SSH 编排
- [ ] 已运行 Python、Swift 和 App 构建基线
- [ ] 已进行真实 UI/Chat 验证
- [ ] 发布前已排除内部文档、凭据和构建产物
