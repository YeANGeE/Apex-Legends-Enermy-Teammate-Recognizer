**语言 / Language：中文 | [English](README.en.md)**

> **声明：本程序 90% 以上内容由 vibe coding + AI Agent 完成，不对兼容性和效果作任何保证，禁止用于非法用途。**

# Apex Legends Enemy / Teammate Recognizer

用于图片和离线视频的敌人 / 队友检测实验项目。当前仍处于 Stage 1 — Local Combat Policy，项目暂停；检测候选不代表完整战斗策略，主动跟随及持续有效射击尚未完成。

## 下载 WebUI

[WebUI v0.1.0（Windows 实验版）](https://github.com/YeANGeE/Apex-Legends-Enermy-Teammate-Recognizer/releases/tag/webui-v0.1.0)

下载并解压 Release 的 ZIP。包内包含检测权重、源码、安装与启动脚本，另需 Python 3.13、NVIDIA CUDA 环境；视频功能需要 FFmpeg。完整使用步骤见 Release 页面和包内 README。这是源码运行包，不是独立 EXE。

## 代码与当前进度

- [全部代码、配置及测试：Training_Files](Training_Files/)
- [完整项目说明](Training_Files/README.md)
- [项目状态](Training_Files/docs/PROJECT_STATUS.md)
- [2026-10-11 暂停检查点](Training_Files/docs/PROGRESS_2026-10-11.md)
- [架构与研究路线](Training_Files/docs/ARCHITECTURE_AND_ROADMAP.md)
- [WebUI 文档](Training_Files/docs/WEBUI.md)

从当前 `main` 克隆后，将 `Training_Files` 作为 PyCharm 项目与运行工作目录，在该目录内安装依赖并放置另外提供的数据和权重。仓库不包含本地录像、标注数据集或模型权重；Release ZIP 单独附带检测权重。

已经发布的 ZIP 与 `webui-v0.1.0` 标签保留发布时的目录结构，不受本次仓库整理影响。
