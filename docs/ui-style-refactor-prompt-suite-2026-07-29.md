# Tokenity UI 风格重构提示词套件 — 2026-07-29

这套提示词用于下一轮 Codex 对 Tokenity macOS SwiftUI 应用做纯视觉重构。

目标是采用八张参考图里的视觉语言，但保留当前已经合理、稳定且通过实机验证的
信息架构、页面布局、交互、状态绑定和后端行为。不要把参考图当成产品规格或
像素级设计稿。

## 使用方法

可以直接复制“总任务提示词”代码块，作为新任务的第一条消息，不需要再拼接旧的
稳定性交接提示词。让 Codex 完成只读审计并建立视觉基线后，再按顺序发送阶段 1–6
的提示词。不要一次复制整份文档或同时让一轮改完所有页面，否则上下文噪声更大，
出现功能回归时也难以定位。

参考图与页面的对应关系：

| 页面 | 参考图 |
|---|---|
| Overview | `/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/01-overview.png` |
| Cluster | `/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/02-cluster.png` |
| Models | `/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/03-models.png` |
| Chat | `/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/04-chat.png` |
| Network / RDMA | `/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/05-network-rdma.png` |
| API Access | `/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/06-api-access.png` |
| Logs | `/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/07-logs.png` |
| Settings | `/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/08-settings.png` |

新 Logo 的唯一源文件：

```text
/Users/zxc/Documents/Tokenity-Stable/Logo-New.png
```

该文件实际为 1254×1254 RGB PNG，没有透明通道。SHA-256：

```text
91462bce59d38f050588512e32a68ed4dd39219095cef05727ab94cf669bd7d4
```

## 总任务提示词

```text
你现在接手 Tokenity 的 macOS SwiftUI 视觉风格重构。

这是一项“视觉重构”，不是产品结构重做、功能重写或后端重构。当前页面布局、
导航结构、数据密度、交互路径和状态模型已经合理，必须保持。八张参考图只用于
提炼视觉风格，不能把图中的示例数据、控件、指标或文案当成新需求。

## 1. 唯一仓库和当前基线

唯一工作仓库：

/Users/zxc/Documents/Tokenity-Stable

不要使用：

/Users/zxc/Documents/MLX-Distributed

开始前执行：

cd /Users/zxc/Documents/Tokenity-Stable
git status -sb
git log -1 --oneline --decorate

当前稳定版提交基线：

2819973200d4d0666aae9d8c69f611159834a752

当前分支：

feature/ui-style-refactor

该分支必须以 `feature/auto-model-router` 的稳定提交
`2819973200d4d0666aae9d8c69f611159834a752` 为父节点。若当前分支或父提交不符，
先停止并报告，不要在其他分支直接实施。

当前工作树有用户保留的未跟踪设计稿、Logo 和交接文件。不要 reset、clean、
checkout、覆盖或删除它们。不要擅自提交或推送。

## 2. 开始前必须完整阅读

1. /Users/zxc/Documents/Tokenity-Stable/docs/live-hardware-validation-2026-07-26.md
2. /Users/zxc/Documents/Tokenity-Stable/apps/TokenityControl/Sources/TokenityControl/Theme.swift
3. /Users/zxc/Documents/Tokenity-Stable/apps/TokenityControl/Sources/TokenityControl/Components.swift
4. /Users/zxc/Documents/Tokenity-Stable/apps/TokenityControl/Sources/TokenityControl/Views.swift
5. /Users/zxc/Documents/Tokenity-Stable/apps/TokenityControl/Sources/TokenityControl/ChatViews.swift
6. /Users/zxc/Documents/Tokenity-Stable/apps/TokenityControl/Sources/TokenityControl/MenuBar.swift
7. /Users/zxc/Documents/Tokenity-Stable/apps/TokenityControl/Sources/TokenityControl/OnboardingView.swift
8. /Users/zxc/Documents/Tokenity-Stable/apps/TokenityControl/Sources/TokenityControl/TokenityControlApp.swift

主入口和状态：

- App 入口：TokenityControlApp.swift
- 页面路由：Views.swift
- Chat：ChatViews.swift
- 全局状态和后端交互：TokenityStore.swift
- 主题：Theme.swift
- 通用组件：Components.swift
- 菜单栏与当前品牌图形：MenuBar.swift

## 3. 参考图

- Overview：/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/01-overview.png
- Cluster：/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/02-cluster.png
- Models：/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/03-models.png
- Chat：/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/04-chat.png
- Network / RDMA：/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/05-network-rdma.png
- API Access：/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/06-api-access.png
- Logs：/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/07-logs.png
- Settings：/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/08-settings.png

先逐张查看参考图，并把观察结果写成一个简短的视觉基线清单，再开始修改。

参考图中的以下内容不能照抄：

- 模型名称、参数量、内存、IP、吞吐、延迟、时间戳和状态。
- 当前产品不存在的图表、筛选器、按钮、统计或交互。
- 与 TokenityStore 当前真实状态不一致的 Ready、Running、Healthy 等状态。
- 参考图里更改过的页面结构、栏目顺序或功能范围。
- 参考图里的旧猫 Logo；新 Logo 源文件才是唯一品牌依据。

## 4. 新 Logo

唯一源文件：

/Users/zxc/Documents/Tokenity-Stable/Logo-New.png

该文件是 1254×1254 RGB PNG，没有透明通道。不得使用 ImageGen 重新生成，
不得重画、改造几何形状、改变字标或换成参考图中的旧 Logo。

先保留源文件原样，再从它派生：

1. 完整品牌锁定版：图形 + Tokenity 字标，用于侧边栏和 Onboarding。
2. 纯图形 mark：只从源图裁切出上方图形，用于 Chat 头像、菜单栏和小尺寸位置。
3. AppIcon：以纯图形 mark 为主体，按 macOS 图标安全区制作；小尺寸不能使用
   完整字标。

源图白色背景不能直接贴到 Dark Mode。应通过确定性的图像处理得到透明背景资源，
或在无法安全去背时为它设计明确的自适应承载底板。必须检查浅色边缘 halo。
不要只按白色像素做粗暴删除而损坏浅蓝抗锯齿边缘。

资源必须进入 App bundle，并由 Swift Package / build script 确定性复制。
不能依赖运行时读取仓库绝对路径。同步更新：

- 侧边栏品牌位
- Onboarding 品牌位
- Chat assistant 头像/空状态中的品牌 mark
- MenuBar mark
- AppIcon.icns

保留所有品牌相关 accessibility label 和 identifier，必要时增加测试。

## 5. 目标视觉语言

整体方向：

- 原生 macOS、安静、精密、编辑感、技术控制台。
- 温暖的象牙白/浅石色，而不是纯白网页后台。
- 深石墨文字、发丝级分隔线、柔和卡片边界。
- Alpine / Cobalt Blue 为唯一主强调色，颜色方向从新 Logo 采样。
- 绿色仅表示正常，橙色仅表示警告，红色仅表示故障或破坏性操作。
- 大面积留白、低噪声、轻层次，避免强阴影和过度装饰。

字体层级：

- 页面标题与主要卡片标题：系统 serif（macOS New York 风格），形成参考图的
  编辑感。
- 正文、导航、按钮、状态、表格：系统 sans-serif（SF Pro）。
- 日志、代码、端点、ID 和技术数值：系统 monospaced。
- 不要给所有文字套 serif；不要引入第三方字体依赖。

建议从 Theme.swift 集中实现，而不是在每个页面散落硬编码：

- Light window：温暖的 off-white。
- Light sidebar：略深的 pale stone。
- Light surface：接近白色但与 window 有轻微层级差。
- Dark window：深石墨，不使用纯黑。
- Dark sidebar：比 window 略深。
- Dark surface：略抬升的 graphite。
- Hairline border：浅灰/深灰语义色，0.5–1 pt。
- Text / secondary / tertiary：使用可访问的语义层级。
- Accent：从新 Logo 的蓝色方向建立 light/dark 对应值。

统一形态：

- 间距基于 4 / 8 / 12 / 16 / 24 / 32。
- 页面外边距、最大内容宽度和当前 NavigationSplitView 宽度保持。
- 卡片圆角建议 12–14，内部控件 7–10，pill 使用 capsule。
- 卡片以细边框和极轻 surface 差异分层；不要给每张卡片大阴影。
- 选中导航使用低饱和浅蓝底 + 蓝色图标/文字。
- 主要按钮使用克制的蓝色填充，次要按钮使用 bordered/outline。
- 禁用态必须清楚但仍可读。
- 状态 pill 尺寸紧凑，不把整行染色。
- SF Symbols 采用一致的线性权重和视觉尺寸。

不要使用：

- 大面积渐变、霓虹、玻璃拟态、强投影。
- Web dashboard 风格的巨型彩色 KPI 卡。
- 为了“像图”而降低真实数据密度。
- 无意义动画或持续动画。
- 在浅色参考图基础上简单反色得到 Dark Mode。

## 6. 布局和功能冻结

必须保持：

- NavigationSplitView、现有 8 个页面和分组顺序。
- Sidebar 宽度范围与窗口最小尺寸。
- Overview、Cluster、Models、Network、API Access、Logs、Settings 的现有
  信息结构和功能。
- Chat 的 header、transcript、composer、history sidebar 基本区域关系。
- 所有按钮 action、Task、Binding、sheet、menu、keyboard shortcut。
- 所有真实状态、状态颜色语义和 disabled 条件。
- Auto Router、resident models、精确 stop/reload、MTP 配置和模型配置。
- Chat 流式更新、Thinking disclosure、停止、重试、复制、编辑、历史、重命名、
  搜索、滚动跟随和 model lock。
- Onboarding、MenuBarExtra、关闭最后窗口不退出、termination cleanup。
- accessibility label、identifier、help 和可键盘操作性。

原则上不要修改 TokenityStore.swift、后端 Python、API 合约或生命周期代码。
若为了展示需要格式化，优先写 View 私有 computed property 或纯 presentation
helper。任何不可避免的状态层修改都必须先说明理由，并用测试证明无行为变化。

不要添加参考图里虚构的数据。没有真实吞吐/延迟时，不显示假值和假曲线。

## 7. 实施顺序

按小步提交式工作，但不要真的 git commit：

1. 只读审计与 before screenshots。
2. Theme token、字体、通用 surface/card/row/button/pill。
3. Logo 资源管线、侧边栏、PageScaffold。
4. Overview + Cluster。
5. Chat。
6. Models + Network。
7. API Access + Logs + Settings。
8. Onboarding + MenuBar + AppIcon。
9. Light/Dark、最小窗口、长文本、空态/加载/失败/禁用态 QA。

每完成一组页面，先构建和运行 Swift 测试，不要把所有问题拖到最后。

## 8. 验证

标准本地基线：

cd /Users/zxc/Documents/Tokenity-Stable
./scripts/verify-stable-baseline.sh

当前最低门槛：

- Python：178 passed，8 skipped。
- Swift：115 passed，0 failed。
- fresh TokenityControl app bundle 构建通过。

UI 构建与启动：

./scripts/run-tokenity-control-app.sh

该脚本会关闭正在运行的开发版 UI，是有状态操作。必须在准备好真实 UI 验证时再运行，
并以脚本打印的 Opened 路径为准。

真实窗口至少验证：

- 8 个页面全部可点击。
- Light 和 Dark。
- 1120×720 最小窗口与更大窗口。
- 长模型名、长 endpoint、空状态、加载、Ready、Warning、Failed、disabled。
- Chat 输入焦点、换行、发送、停止、滚动、历史展开/收起。
- Settings/Help 重新打开 Onboarding。
- 菜单栏图标在不同状态下可辨认。
- 新 Logo 在浅色/深色、大尺寸/小尺寸下无白框、halo、模糊和裁切。

如果 Computer Use 可用，用真实 UI 完成点击和截图验证。SwiftUI smoke test
不能替代真实窗口检查。

纯视觉任务不需要重启两台 Mac、加载模型或修改生产 9100 Agent。不要执行
stop-all，不要部署后端，不要改变当前集群状态。

## 9. 交付

完成后提供：

1. 修改文件清单。
2. 视觉 token 与组件变化摘要。
3. Logo 资源派生方式与 SHA。
4. 每个页面的 before/after 截图，Light/Dark 至少各一套。
5. Python、Swift、App bundle 测试结果。
6. 明确说明功能、状态绑定和后端未改变。
7. 未解决的视觉问题或无法可靠验证的状态。

先完成只读审计、参考图视觉基线和 Logo 资源计划，汇报后再修改代码。
```

## 阶段 1：视觉系统、Logo 与应用外壳

```text
继续 Tokenity UI 风格重构的阶段 1。只处理设计系统、品牌资源和应用外壳，不改
具体业务页面的数据或交互。

目标：

1. 扩展 Theme.swift：
   - light/dark 的 window、sidebar、surface、raised surface、border、
     separator、text、secondary、tertiary、accent、success、warning、danger、
     code surface。
   - 建立 display serif、section serif、body sans、caption、mono 的字体 helper。
   - 所有颜色有明确语义，不在页面里散落重复 RGB。

2. 扩展 Components.swift：
   - 保留现有 StatusPill、InfoGroup、InfoRow、MemoryUsageBar、CodeBlock、
     OperationLog、PageScaffold 的 API 或提供兼容 wrapper。
   - 统一卡片边框、圆角、row separator、按钮和状态样式。
   - 不能因视觉重构破坏现有调用者或改变业务布局。

3. 处理新 Logo：
   - 源文件：
     /Users/zxc/Documents/Tokenity-Stable/Logo-New.png
   - 保留源文件，确定性派生 lockup、mark 和 AppIcon。
   - 不用 ImageGen，不改画。
   - 解决无 alpha 与 Dark Mode 白底问题。
   - 资源打进 App bundle，不能运行时读取绝对路径。

4. 重构 SidebarView 和 PageScaffold：
   - 保留 NavigationSplitView、侧边栏宽度、导航分组与页面顺序。
   - 参考图的 pale stone sidebar、soft-blue selection、细线分隔和编辑感标题。
   - 品牌位使用新 Logo，但不要为了模仿图而显著改变侧边栏占宽或挤压导航。

5. 同步 Onboarding、Chat 品牌 mark、MenuBar mark 和 AppIcon 的资源接口，但本阶段
   不重排它们的内容。

验收：

- 现有页面无需逐页修改即可编译。
- Light/Dark 都没有硬编码白底。
- 小尺寸 mark 清晰，完整 lockup 不被压扁。
- accessibility identifier 保持。
- Swift 115 passed / 0 failed 或更多。
- App bundle 构建后资源实际存在于 Contents/Resources。

完成后先展示侧边栏、空白 PageScaffold、Onboarding 品牌位和菜单栏 mark 的
Light/Dark 截图，再进入业务页面。
```

## 阶段 2：Overview 与 Cluster

```text
继续 Tokenity UI 风格重构的阶段 2，只处理 OverviewPage、ClusterPage 以及它们
直接使用的纯展示子视图。

参考：

- /Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/01-overview.png
- /Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/02-cluster.png

约束：

- 保留 Overview 当前 Cluster State、Resident Models、Selected Macs、Readiness
  的信息与顺序。
- 保留 Cluster 当前 Cluster Builder、Cluster Setup、advanced setup、
  acceleration、plan、所有按钮和 disabled 条件。
- 不修改 TokenityStore、节点选择、创建/停止、RDMA 或 MTP 逻辑。
- 不添加参考图里的假拓扑、假吞吐或假内存。

风格目标：

- 大但克制的 serif 页面标题和 section 标题。
- 卡片采用低对比 surface + hairline border，不使用重阴影。
- Overview 的状态信息保持清晰的表格/行结构；可以通过更好的列对齐、留白和
  紧凑 pill 提升层级，但不能隐藏真实字段。
- Cluster node card 更接近参考图的技术节点卡：细边框、明确选中态、轻量连接线、
  coordinator/worker 层级清楚。
- selected、online、blocked、warning、failed 在 Light/Dark 中都可辨识，
  不能只靠颜色。
- 长 IP、节点名、readiness issue 不截断到无法理解。

验证：

- 节点点击选择行为不变。
- canEditCluster、Create、Refresh、Stop 的启用条件不变。
- 最小窗口不出现横向裁切。
- Overview/Cluster Light 与 Dark 截图。
- 运行 Swift 测试和相关视觉 smoke tests。
```

## 阶段 3：Chat

```text
继续 Tokenity UI 风格重构的阶段 3，只处理 ChatViews.swift 中的视觉层。

参考：

/Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/04-chat.png

Chat 是最高风险页面。当前功能和布局行为已经稳定，禁止重写消息流、滚动状态机、
NSTextView bridge 或历史持久化。

必须保持：

- 左侧主导航、中央 Chat、右侧可收起 History 的三段关系。
- ChatWorkspaceHeader 的标题、路由/模型/连接状态、新会话和 History toggle。
- ChatTranscriptFollowState 和 TranscriptScrollPositionObserver。
- streaming token coalescing、自动跟随、用户手动滚动后停止跟随、Latest 按钮。
- user/assistant/thinking/error/stopped 消息状态。
- Thinking disclosure、Markdown、代码块、复制、重试、route detail、编辑。
- composer 原生输入、输入焦点、Return/Shift-Return、send/stop、Auto/手动模型、
  Fast/Balanced/Quality、model lock。
- history 搜索、选择、新建、重命名、删除和时间分组。

风格目标：

- 参考图的 editorial Chat：宽松正文、清晰用户消息、克制 assistant surface、
  右侧浅石色 History。
- Header 使用 serif 会话标题，其余元数据保持 sans/mono。
- 用户消息用轻蓝描边或浅蓝 surface；assistant 不做沉重聊天气泡。
- Thinking 区域应是轻量 disclosure，而不是巨型彩色卡。
- composer 保持当前位置和功能，改成细边框、柔和层次、清晰 focus ring。
- send/stop 是唯一高强调动作；其他操作退到 outline/tertiary。
- 流式更新时不能因 shadow、material、geometry 或动画引入 layout feedback loop。
- History 选中项使用与主 Sidebar 一致的浅蓝选择语言。

禁止：

- 用 List/ScrollView 大改替换现有滚动实现。
- 删除 NSViewRepresentable 输入或滚动观察器。
- 为“好看”隐藏 metrics、route provenance、停止或错误状态。
- 在每个 token 上启动动画、阴影或昂贵 blur。

验证：

- 现有 Chat、Parsing、Markdown、VisualSmoke 全部通过。
- 10,000 次 scroll transition 测试仍通过。
- 真窗口检查输入、换行、发送、停止、流式、手动滚动、Latest、History。
- Light/Dark、空对话、短回答、长 Markdown、代码块、Thinking、错误态截图。
```

## 阶段 4：Models 与 Network / RDMA

```text
继续 Tokenity UI 风格重构的阶段 4，处理 ModelsPage、NetworkPage 及其纯展示子视图。

参考：

- /Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/03-models.png
- /Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/05-network-rdma.png

Models 必须保持：

- Model Library、resident model pool、available models 的当前结构。
- Scan、Configure、Load、Stop、Use in Chat。
- Allow Auto、Keep Resident。
- model status、active request、capability、MTP、quantization、memory、instance
  provenance 和失败信息。
- 精确 instance stop/reload，不得退回全局清理。
- 所有 disabled 条件和加载中身份保护。

Network 必须保持：

- 当前 readiness、节点属性、network plan 与真实 Agent 数据。
- RDMA/Standard Network 的真实状态和错误。
- 不添加参考图里没有数据来源的 throughput、latency 或 sparkline。

风格目标：

- 保留当前紧凑数据密度，以参考图的细线表格、serif section heading、蓝/绿状态
  和轻量 memory bar 统一视觉。
- resident pool 与 available models 的列必须稳定对齐。
- 长模型名用合理 truncation + help/tooltip，不能丢失辨识度。
- Configure/Load/Stop 有清晰主次；危险动作只使用红色文字/轻底，不大面积红色。
- loading、ready、failed、not loaded、stopping 在 Light/Dark 中差异明确。
- Network 用现有真实数据建立连接关系层级，不制造仪表盘。

验证：

- ResidentModelRecoveryTests 全部通过。
- 加载中的新实例不能被旧 snapshot 覆盖。
- UI 长模型名、两个 resident、失败行、MTP 不支持、低内存警告都能显示。
- Models/Network Light 与 Dark 截图。
```

## 阶段 5：API Access、Logs 与 Settings

```text
继续 Tokenity UI 风格重构的阶段 5，处理 APIAccessPage、LogsPage、SettingsPage。

参考：

- /Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/06-api-access.png
- /Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/07-logs.png
- /Users/zxc/Documents/Tokenity-Stable/docs/ui-style-reference-2026-07-29/08-settings.png

API Access：

- 保留当前真实 endpoint、模型、示例、复制和说明。
- endpoint、HTTP method、JSON、curl 使用 monospaced。
- 复制按钮反馈必须保留。
- 不添加尚未实现的 Test Connection、API key 或 endpoint。
- 可用性状态只能来自当前真实状态。

Logs：

- 保留当前日志数据源和 OperationLog 能力。
- 不添加没有实现的数据级别筛选、导出、event counter 或 follow latest。
- 日志正文使用 monospaced；时间、level、source 只有在真实结构化数据存在时才分列。
- 大量日志滚动仍应轻量，避免每行 material/shadow。

Settings：

- 保留当前 Console、Backend Status、Getting Started 和 Welcome guide 入口。
- 参考图中的 Appearance、Accent Color 目前没有对应的产品状态，不能为了像图而添加
  假设置；只有用户另行提出功能需求后才能实施。
- 不增加任何没有状态支持的设置。
- 表单行使用参考图的安静分组、hairline separator 和清晰尾部控件。
- App 继续响应 macOS 系统 Light/Dark 与系统 accent，不新增应用内持久化开关。

统一风格：

- 主要分组用 serif heading；row label、控件、说明保持 sans。
- 代码面板使用轻微不同 surface 与细边框，不使用纯黑终端样式。
- copy/edit/folder 等图标按钮保持一致尺寸和 focus ring。

验证：

- API copy 与 Settings 的 Show Guide 等现有实际操作正常。
- 长 endpoint 和长日志不破坏布局。
- 三个页面的 Light/Dark 截图。
- Swift 测试全部通过。
```

## 阶段 6：Onboarding、MenuBar、全局 QA 与交付

```text
完成 Tokenity UI 风格重构的阶段 6。

1. Onboarding：
   - 保留页数、步骤、按钮、完成状态与 Settings/Help 重开逻辑。
   - 使用同一视觉 token 和新 Logo lockup。
   - 不重写 Runtime Bootstrap，不改变安装行为。

2. MenuBar：
   - 使用新 Logo 的纯图形 mark，但保持不同运行状态的小状态点和 tooltip。
   - 16–20 pt 下必须清晰；不能放完整 Tokenity 字标。
   - 保持 MenuBarExtra 内容和状态映射。

3. AppIcon：
   - 从同一 Logo mark 派生，不重新生成。
   - 更新 Resources/AppIcon.icns。
   - 验证 build script 将它复制到 App bundle，Info.plist 指向正确。

4. 全局 QA：
   - 对照八张参考图确认视觉语言统一，但不做像素复刻。
   - 逐页 Light/Dark。
   - 1120×720、1280×820 和更大窗口。
   - hover、pressed、focused、selected、disabled、loading、warning、failed。
   - keyboard navigation、VoiceOver label、help、identifier。
   - 检查所有 Logo 场景无白框、halo、模糊、裁切。

5. 完整验证：

cd /Users/zxc/Documents/Tokenity-Stable
./scripts/verify-stable-baseline.sh

必须至少保持：

- Python 178 passed / 8 skipped。
- Swift 115 passed / 0 failed。
- App bundle build passed。

然后使用：

./scripts/run-tokenity-control-app.sh

在真实 UI 中完成 8 页点击和截图。不要重启远端 Mac，不要加载模型，不要修改
生产 9100 Agent。

6. 交付报告：
   - 修改文件。
   - 主题 token。
   - Logo 派生资产及 SHA。
   - 八页 Light/Dark 截图索引。
   - 测试结果。
   - 功能和后端未改变的证据。
   - 已知限制。

不要自动 commit、push、安装生产 App 或重建发布 DMG，除非用户随后明确要求。
```

## 视觉验收速查表

下一轮完成后，可用以下清单做最后核对：

- [ ] 参考图只影响视觉语言，没有把示例数据复制进产品。
- [ ] 页面、栏目、导航、按钮与功能没有被删除或重排。
- [ ] TokenityStore、Python 后端和 API 合约没有因纯视觉工作发生行为变化。
- [ ] 新 Logo 来自 `Logo-New.png`，没有重新生成或改画。
- [ ] Sidebar、Onboarding、Chat、MenuBar、AppIcon 已全部替换。
- [ ] Logo 在 Dark Mode 没有白色矩形和抗锯齿 halo。
- [ ] 页面/section 标题使用 serif，正文仍是易读的 SF sans。
- [ ] 日志、代码、endpoint、ID 使用 monospaced。
- [ ] 颜色、字体、圆角和间距集中在主题/组件，不散落硬编码。
- [ ] Light/Dark、最小窗口、长文本和所有状态通过。
- [ ] Chat 滚动、输入、流式和 History 没有回归。
- [ ] Models 加载身份、resident sibling 和精确停止没有回归。
- [ ] Python、Swift、App bundle 达到或超过当前基线。
- [ ] 已完成真实 App 八页点击验证，而不是只看 snapshot。
