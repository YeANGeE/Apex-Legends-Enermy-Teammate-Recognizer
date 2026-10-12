**语言 / Language：中文 | [English](https://github.com/YeANGeE/Apex-Legends-Enermy-Teammate-Recognizer/blob/main/README.md)**

> **声明：本程序 90% 以上内容由 vibe coding + AI Agent 完成，不对兼容性和效果作任何保证，禁止用于非法用途。**

# Apex Legends Enemy / Teammate Recognizer

这是一个用于检测图像和离线视频中敌人和队友的实验性项目。该项目是专精FPS游戏的陪玩游戏AI（Companion Gaming AI, CGA）项目的子项目，旨在为缺少队友的玩家提供陪玩功能。目前项目暂停于第一阶段——局部战斗策略。检测结果尚不足以构成完整的战斗策略；主动目标跟踪和持续有效射击等功能仍未完成。未来的阶段将包括完整角色战斗策略模型、角色拾取行为模型、整体战斗模型以及队友语音指令识别模型。

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
