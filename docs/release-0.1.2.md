# Tokenity v0.1.2 — GLM 5.3 · DeepSeek V4 Flash · Qwen3.8-Next · MiniMax H3 Turbo

**版本：0.1.2｜Build：4｜Runtime：2026.09.15.2｜Apple silicon｜macOS 26.2+**

本版重点增加 **GLM 5.3 Flash、DeepSeek V4 Flash、Qwen3.8-Next / Flash-Next**
模型兼容性，以及 **MiniMax H3 Turbo 视频加速**。发布版 PKG 已完成 Developer ID
签名、Apple 公证和公证票据附加，并在另一台 M5 Ultra 上完成安装与基础功能验证。

## 支持的模型

| 模型 | 后端与本版支持 |
| --- | --- |
| **GLM 5.3 / GLM-5.3-Flash** | `glm5_next`，自动使用独立 MLX-VLM 后端；单 Mac 文本对话、流式输出及兼容的图片输入 API。 |
| **DeepSeek V4 Flash** | `deepseek_v4`，MLX-LM 兼容层；支持已验证的 Flash / Flash-4bit MLX checkpoint，处理混合 MXFP4 expert 元数据和缺失的文本聊天模板。 |
| **Qwen3.8-Next / Qwen3.8-Flash-Next** | `qwen4_exp`，自动使用独立 MLX-VLM 后端；单 Mac 运行，保留 OpenAI 兼容接口。 |
| **MiniMax H3** | 原生视频后端；标准 28 步，以及 Turbo **4 / 6 / 8 步**档位。Turbo 默认 6 步、strength 1.0，目前限单 Mac；标准模式保留单机和双机 TP2/RDMA。 |
| **Qwen3.5 系列** | 继续支持兼容的 dense / MoE MLX checkpoint，包括 Qwen3.5-9B 和 Qwen3.5-122B-A10B。 |
| **GLM 5.2** | 保留已有 MLX-LM 兼容路径和已验证的双机 JACCL/RDMA 推理工作流。 |

模型识别依据 `config.json` 和后端能力；表中的支持针对兼容的 MLX checkpoint，
不代表任意量化格式、任意内存配置或所有模型都支持多机运行。模型权重需单独准备。

## 本版改进

- MLX-LM 与 MLX-VLM 使用独立运行环境，按模型架构自动选择，避免依赖版本冲突。
- DeepSeek V4 使用适配其混合压缩 KV cache 的顺序生成路径；兼容修正不改写用户模型文件。
- H3 Turbo 提供不同采样步数，便于权衡生成速度与效果；保留视频预览、导出和历史记录。
- 修复签名打包过程中公开资源和运行环境符号链接权限过严的问题，普通用户安装后可以正常读取和启动。
- 完整 PKG 内置 SwiftUI 应用、Node Agent、Watchdog、Python、MLX-LM、MLX-VLM、H3 原生组件和离线修复包。
- Runtime 升级先校验再替换，保留原有模型目录，并在健康检查失败时恢复上一份 Runtime。

## 安装

从 [v0.1.2 Release](https://github.com/HeyZhey/Tokenity/releases/tag/v0.1.2)
下载 **Tokenity-0.1.2-macos-arm64.pkg**，通过 macOS 安装器按提示输入本机管理员密码。
安装后从“应用程序”打开 Tokenity。目标 Mac 不需要额外安装 Xcode、Homebrew 或 Python。

发布包已通过 Gatekeeper 安装检查，无需关闭 Gatekeeper 或 SIP。
多机使用时，每台参与计算的 Mac 都安装同一版本；模型放在 `/Library/Tokenity/Models`
或通过应用选择目录。H3 Turbo 还需匹配的 adapter，见 [Turbo 说明](tokenity-video-gen-turbo.md)。

## 验证范围

最终 PKG 在 **M5 Ultra / 256 GiB / macOS 27.0** 上完成：

- 标准安装器升级、下载隔离标记下的 Gatekeeper 检查、应用签名和启动检查。
- 普通用户资源访问和全部 17 个 Runtime 符号链接解析；重装后无需手动修改权限。
- MLX-LM、MLX-VLM 的 GPU 运算，以及 Qwen3.5-9B、Qwen3.8-Flash-Next 的实际短文本推理。
- DeepSeek V4、GLM 5.3、Qwen3.8-Next 兼容模块导入。
- H3 服务启动、健康检查和 Turbo 就绪检查；原有模型目录完整保留。

本次最终包验收是安装和功能测试，没有重跑 DeepSeek / GLM 大模型加载、长上下文 benchmark、
图片输入或 H3 视频生成。此前模型验证及限制见
[DeepSeek V4](deepseek-v4-compat.md)、[MLX-VLM](mlx-vlm.md) 和 [H3 视频](minimax-h3-video.md)。

## 使用边界

- MLX-VLM 接入目前限单 Mac；桌面聊天输入仍为文本，图片可通过 API 的 base64 `image_url` 提交。
- H3 Turbo 目前限单 Mac；不要将 Turbo 档位理解为已支持 TP2。
- H3 视频常驻服务在应用重启后需要重新点击 **Start Video Runtime**。
- 大模型和长上下文需要足够统一内存，加载与上下文长度仍受内存准入检查约束。

## Build information

The **v0.1.2 (Build 4)** release adds GLM 5.3 Flash, DeepSeek V4 Flash,
Qwen3.8-Next / Flash-Next, and MiniMax H3 Turbo. The published arm64 PKG is
Developer ID signed, Apple notarized, and stapled. It has passed installation
and short-inference checks on a separate M5 Ultra.

The source command `scripts/package-tokenity-dmg.sh` builds the development
package; release signing and notarization are separate steps. Runtime versions,
H3 protocol 1, Turbo protocol 1, 259 adapter modules and strength 1.0 are pinned
by the runtime lock. See [installer documentation](installer-dmg.md).
