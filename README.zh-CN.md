<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/TokenityControl/Sources/TokenityControl/Resources/TokenityBrandLockupDark.png">
    <source media="(prefers-color-scheme: light)" srcset="apps/TokenityControl/Sources/TokenityControl/Resources/TokenityBrandLockup.png">
    <img alt="Tokenity" src="apps/TokenityControl/Sources/TokenityControl/Resources/TokenityBrandLockup.png" width="220">
  </picture>
</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

# Tokenity

Tokenity 将局域网内的 Apple 芯片 Mac 组成一个可管理的 AI 集群。它以一套原生
控制面统一管理语言和视频推理，并将分布式执行、资源隔离、推理加速与故障恢复
纳入运行时。

项目包含：

- **TokenityControl** —— 原生 SwiftUI 应用，负责集群发现、拓扑、模型管理、
  对话、视频生成、健康状态、日志与恢复。
- **Tokenity Node Agent** —— 部署在每台推理 Mac 上的受控 HTTP 服务。
- **Tokenity runtimes** —— 通过统一 OpenAI 兼容网关提供分布式 MLX 推理和
  针对具体工作负载的后端。

## 产品特性

- **Mac 集群控制面** —— 在一个原生 macOS 应用中发现、选择、检查并管理
  Apple 芯片节点。
- **局域网自动发现** —— 自动发现私有网络中的 Node Agent，使用稳定
  `machine_id` 识别 Mac，并在地址变化后修复连接，远程节点绝不退回 loopback。
- **高速分布式数据面** —— 优先使用 Thunderbolt RDMA/JACCL，并提供严格的
  readiness 检查；其他部署可使用标准网络模式。
- **推理加速** —— 基于模型元数据判断兼容性，检测 Native MTP 能力并安全启用，
  不依赖写死的模型名称。
- **统一工作负载** —— 同时支持 OpenAI 兼容的流式/非流式 LLM 推理和原生视频
  生成；MiniMax H3 是当前已经验证的视频后端，而不是产品边界。
- **完整运行周期** —— 多实例隔离、自动路由、请求排队、超时、取消、内存准入、
  资源账本、健康/readiness 探测、watchdog 和重启恢复。
- **集群安全操作** —— 使用类型化 HTTP 编排、协调的 rank 启停，产品路径不依赖
  SSH。

## 架构

```text
                    ┌──────────── HTTP 控制面 ────────────┐
TokenityControl ───►│ Coordinator Agent     Worker Agents │
局域网自动发现 ────►│ 健康 · 路由            本地 watchdog │
                    └──────────────┬──────────────────────┘
                                   │ 稳定网关 :9100
OpenAI 客户端 ──────────────────────┤
                                   ▼
                              LLM · 视频工作负载
                               ╲              ╱
                            Thunderbolt RDMA/JACCL
                                或标准网络
```

Node Agent 只启动和监管本机进程。分布式 rank 通过类型化 HTTP 请求协调，模型
collective 使用所选择的数据面。产品不使用 SSH。当前真实硬件验证覆盖单节点和
双节点工作负载，但控制面基于 Agent，不是固定的双机拓扑。

## 运行要求

- 位于同一可信网络的 Apple 芯片 Mac。
- Tokenity App 开发需要 macOS 26.2+ 和 Swift 5.9+。
- 后端开发需要 Python 3.10+。
- 每台推理 Mac 上安装兼容的 MLX/MLX-LM Runtime。
- 分布式工作负载使用的模型在所有参与节点上具有相同路径。
- TokenityControl 能够访问每个 Node Agent 的 TCP `9100` 端口。
- RDMA 模式需要直连且活动的 Thunderbolt 链路、RDMA 设备和对端 IP。

当前已验证的打包 Runtime 仅支持 Apple 芯片，并要求 macOS 26.2 或更高版本。
模型权重不包含在项目中。

## 开发环境

```bash
git clone <repository-url> Tokenity
cd Tokenity

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

构建并打开原生应用：

```bash
./scripts/run-tokenity-control-app.sh
```

启动开发用 Node Agent：

```bash
tokenity node-agent --host 0.0.0.0 --port 9100
```

持久部署请在每台推理 Mac 上安装 Node Agent 包，参见
[安装与打包](docs/installer-dmg.md)。

## 基本流程

1. 在每台 Mac 上安装 Node Agent，然后打开 **Cluster**。
2. 由局域网发现自动找到节点，或直接连接 Agent。
3. 选择参与工作负载的 Mac，确认数据面显示 **Ready**。
4. 创建集群拓扑。
5. 打开 **Models**，输入各节点共有的绝对模型目录，然后选择
   **Scan Models**。
6. 加载模型，通过 **Chat** 或外部 API 开始推理。
7. 打开 **Video**，检查视频 Runtime、生成视频、查看流式进度、取消任务、
   预览并导出产物。

LLM 模型目录与视频 Runtime 路径是独立设置。只有位于扫描目录中、且存在于
所选 Mac 上的模型才会出现在模型列表中。

## OpenAI 兼容 API

Coordinator 通过以下地址提供稳定网关：

```text
http://<coordinator-host>:9100/v1
```

主要接口：

- `GET /v1/models`
- `GET /v1/gateway/routes`
- `POST /v1/chat/completions`
- `POST /v1/video/generations`

示例：

```bash
export TOKENITY_HOST=<coordinator-host>

curl "http://${TOKENITY_HOST}:9100/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<model-id>",
    "messages": [{"role": "user", "content": "Hello from Tokenity"}],
    "stream": false
  }'
```

默认不启用身份认证和 TLS，请仅在可信局域网内开放 API。

## 验证

运行完整的本地稳定基线：

```bash
./scripts/verify-stable-baseline.sh
```

该脚本执行 Python 测试、Swift 测试，并全新构建 TokenityControl debug app。
真实的分布式模型、视频、MTP 和 RDMA 验证还需要对应 Runtime、checkpoint 和
集群硬件。当前硬件基线包括 GLM 5.2 和 MiniMax H3 TP2。

## 打包

```bash
./scripts/build-tokenity-control-app.sh
./scripts/package-tokenity-dmg.sh
```

发布打包、Runtime 固定、签名状态和 Node Agent 安装参见
[安装与打包](docs/installer-dmg.md)。

## 文档

- [部署配置](docs/deployment-configuration.md)
- [HTTP Node Agent](docs/http-node-agent.md)
- [外部 API](docs/external-api.md)
- [GLM 5.2 兼容性](docs/glm-5.2-compat.md)
- [MiniMax H3 视频](docs/minimax-h3-video.md)
- [Native MTP](docs/native-mtp.md)
- [RDMA 操作](docs/current-usage-rdma-qwen.md)
- [稳定基线](docs/stable-baseline.md)

## 仓库结构

```text
apps/TokenityControl/  SwiftUI 应用和测试
tokenity/              Python CLI、Node Agent、路由和运行时
tests/                 Python 回归测试
scripts/               构建、验证、部署和打包工具
docs/                  面向操作与兼容性的专题文档
```

## 许可

Tokenity 尚未声明开源许可证，保留所有权利。采用独立许可证的代码参见
[第三方声明](THIRD_PARTY_NOTICES.md)。
