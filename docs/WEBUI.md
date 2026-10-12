# 图片与视频检测 WebUI

## Release 下载包

Windows 源码运行包随附当前检测权重，下载地址为仓库 GitHub Releases。
安装和启动流程见 `packaging/webui/README.md`。发布包使用 `--no-gallery`
关闭依赖作者本地标注集的队友样本筛选；图片和视频检测入口保留。
`python build_webui_release.py` 可从已提交的工作区生成 ZIP 和 SHA-256 文件，
仅打包文件，不启动服务、训练或推理。

当前默认模型：`runs/c1_yolo/day4_yolo26n_1280_joint_supplement_v1/weights/best.pt`。
这是项目检查点选用的 C1 检测权重；视频功能不输出手柄动作。

在 PyCharm 中运行 `c1_webui.py`，解释器为项目内
`.venv-yolo/Scripts/python.exe`，工作目录为项目根目录，无需参数。
浏览器地址为 `http://127.0.0.1:7860`。
服务启动和停止均由用户管理。如果旧服务已占用端口，先停止旧 WebUI。
桌面 `YOLO 敌人检测` 快捷方式保留启动入口，但不会自动替换正在运行的旧版服务。

视频依赖在 `.venv-yolo` 中安装：`pip install -r requirements-webui.txt`。
还需要现有 FFmpeg；项目使用系统 PATH 或 WinGet Links 查找它。

## 使用视频功能

1. 页面下方选择视频，设定上方置信度阈值，点击“生成带框视频”。
2. 处理期间显示当前带框画面、帧数和进度。完成后可播放及下载 MP4。

最大上传 2 GB，一次处理一个视频；可取消上传或生成。
逐帧独立检测敌人/队友，不建立持续目标 ID。
这是离线视频生成，预览按处理进度更新，不保证以原视频速度实时推理。
输出 H.264 MP4，音轨（如有）转为 AAC。帧播放时间由源视频 PTS 保留，
不以固定帧率替换变帧率时间轴。奇数尺寸在右侧或底部补一个黑色像素，
以兼容浏览器常用的 YUV420 编码。缺失或不递增的时间戳会明确报错。

结果在 `c1_webui_outputs/videos/<任务ID>/`：

- `result.mp4`：带中文类别与置信度的结果视频。
- `detections.jsonl`：每帧原始 PTS、时间基准及检测框。
- `summary.json`：模型权重路径、阈值和处理结果。
- `preview.jpg`：最近显示的处理画面。

生成完成或失败后清理上传的视频副本及中间视频；原文件和数据集不变。
结果播放前自动完整解码，核对导出帧数。界面耗时不作为模型性能基准。
