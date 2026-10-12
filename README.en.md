**Language / 语言: [中文](README.md) | English**

> **Notice: More than 90% of this program was created through vibe coding + AI Agents. No guarantees are made regarding compatibility or effectiveness. Use for illegal purposes is prohibited.**

# Apex Legends Enemy / Teammate Recognizer

An experimental project for detecting enemies and teammates in images and offline videos. The project is currently paused at Stage 1 — Local Combat Policy. The detection candidates do not constitute a complete combat policy; active target following and sustained effective firing remain unfinished.

## Download the WebUI

[WebUI v0.1.0 (Experimental Windows Release)](https://github.com/YeANGeE/Apex-Legends-Enermy-Teammate-Recognizer/releases/tag/webui-v0.1.0)

Download and extract the release ZIP. It includes detection weights, source code, dependency installation scripts, and startup scripts. Python 3.13 and an NVIDIA CUDA environment are required separately; video processing also requires FFmpeg. Full instructions are available on the release page and in the README included in the ZIP. This is a source package that runs through Python, rather than a standalone EXE.

## Source Code and Current Progress

- [All code, configuration, and tests: Training_Files](Training_Files/)
- [Full project documentation](Training_Files/README.md)
- [Project status](Training_Files/docs/PROJECT_STATUS.md)
- [Paused checkpoint dated 2026-10-11](Training_Files/docs/PROGRESS_2026-10-11.md)
- [Architecture and research roadmap](Training_Files/docs/ARCHITECTURE_AND_ROADMAP.md)
- [WebUI documentation](Training_Files/docs/WEBUI.md)

After cloning the current `main` branch, use `Training_Files` as both the PyCharm project directory and the working directory for execution. Install dependencies there and place separately supplied datasets and weights in that directory. The repository does not include local recordings, annotation datasets, or model weights; detection weights are included separately in the release ZIP.

The published ZIP and the `webui-v0.1.0` tag retain the directory structure used at release time and are unaffected by the subsequent repository reorganization.
