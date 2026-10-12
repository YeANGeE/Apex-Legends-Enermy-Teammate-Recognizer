> **声明：本程序 90% 以上内容由 vibe coding + AI Agent 完成，不对兼容性和效果作任何保证，禁止用于非法用途。**

# 敌人 / 队友检测 WebUI v0.1.0

Windows 源码运行包，包含 YOLO26n Day4 检测权重、图片检测与离线视频检测界面。
这不是独立 EXE，也不包含 Python、PyTorch、CUDA 驱动或 FFmpeg。
首次安装需要网络及数 GB 可用空间。未在其他电脑完成兼容性验证。

## 安装与启动

1. 安装 Windows x64 Python 3.13（含 Python Launcher，即 `py` 命令）。
   需要 NVIDIA CUDA 显卡和支持 CUDA 13.2 的驱动；不支持 CPU-only、AMD 或 macOS。
   开发环境使用 RTX 5090，不承诺其他显卡的性能或兼容性。
2. 解压整个 ZIP 到可写目录，双击 `Install-Dependencies.cmd`。
   脚本创建包内 `.venv-yolo`，下载固定版本的 PyTorch、Ultralytics 等依赖。
   安装结束只检查 CUDA，不启动 WebUI，也不进行模型推理。
3. 视频功能另外需要 FFmpeg 和 ffprobe，加入 PATH。
   安装后在终端执行 `ffmpeg -version` 和 `ffprobe -version` 确认能找到程序。
4. 双击 `Start-WebUI.cmd`，浏览器打开 `http://127.0.0.1:7860`。
   初次启动会加载模型并预热。保留命令窗口，按 Ctrl+C 停止服务。
   端口被占用时，先停止占用该端口的程序。

也可以在 PyCharm 选择 `.venv-yolo/Scripts/python.exe`，运行 `c1_webui.py`，
参数 `--no-gallery`，工作目录为解压后的包目录。

输出保存到包内 `c1_webui_outputs/`。视频最大 2 GB，一次处理一个。
本包关闭依赖作者本地标注集的队友样本筛选功能；不包含录像、标注集、
训练服务、手柄控制、自动瞄准或自动射击入口。模型只是检测候选，不代表完整战斗策略。

## 模型、源码与检查

权重路径：`runs/c1_yolo/day4_yolo26n_1280_joint_supplement_v1/weights/best.pt`。
随附训练 `results.csv` 用于界面显示最佳 epoch；数据集不在包内。
`MANIFEST.json` 记录源码提交和每个文件的 SHA-256；Release 随附 ZIP 校验文件。

完整源码与训练代码：
https://github.com/YeANGeE/Apex-Legends-Enermy-Teammate-Recognizer/tree/webui-v0.1.0

依赖 Ultralytics 的许可证为 AGPL-3.0，完整文本随附于
`third_party/ULTRALYTICS-LICENSE.txt`。其他依赖由安装脚本下载，保留其各自许可证。
声明不替代第三方许可证。
