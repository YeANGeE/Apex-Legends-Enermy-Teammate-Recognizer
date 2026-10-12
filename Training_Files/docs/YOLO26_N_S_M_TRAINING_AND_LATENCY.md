# YOLO26 n/s/m联合训练与最后统一延迟对比

阶段：Stage 1 — Local Combat Policy。
三者从各自通用预训练权重起步，不是随机权重训练。
都用1384训练图、62旧验证图、1280输入、batch=4、AdamW、lr0=0.001、
warmup_bias_lr=0.001、最多60轮、patience=20、同一seed及增强设置。
模型规模及对应通用预训练权重不同，输出目录各自独立。

## 用户依次运行

当前n运行继续完成，不停止或修改这个进程。
随后在PyCharm依次运行：

1. `train_c1_yolo26s_joint_pretrained.py`
2. `train_c1_yolo26m_joint_pretrained.py`

两者参数留空，解释器为 `.venv-yolo/Scripts/python.exe`，工作目录FPS根目录。
预训练权重已下载，检查已通过，不需要用户安装或重复check-only。
出现OOM时保留完整输出，交给Codex准备调整；不要同时更改batch和分辨率。
正常结束或早停都记录实际best.pt及结果目录；best来自验证选中的轮次，不一定最后一轮。
n/s/m及其他候选全部训练结束之后才测速，不同时运行多个GPU训练/测速。

## 模型路径登记

`c1_all_model_comparison.json`预置7个模型：原V4、Day3微调、偏置预热对照、联合n/s/m、n完整60轮对照。
如果Ultralytics自动递增目录名，以控制台实际路径为准；用户发结果后Codex更新登记。
工具不默默跳过缺失模型，也不拿仍在训练中的中间checkpoint测速。
联合n/s/m核对数据哈希、初始权重和一致的训练设置。

## 识别比较

全部准备好后，用户在PyCharm运行 `compare_c1_all_model_accuracy.py`，参数留空。
统一27张外部诊断图，统一COCO指标和0.05/0.25阈值，输出原尺寸PNG拼接。
这批图片已经反复检查，不能作为独立泛化证明。最终替换仍需新比赛验收。
多余框、框松紧、真实背景误检和身份错误分别看图解释。

## 延迟比较

运行 `benchmark_c1_all_models.py`；先参数`--check-only`检查完整登记和图片。
检查通过后参数留空，即统一检测计时。

默认每模型3组随机顺序测试，每组预热50次、正式300次，batch=1、FP16、
1280方形输入、相同NMS分支（nms=None且运行时end2end=False）。
每个模型使用相同的27张1920×1080本机录制图片，图像提前解码，
这些图片仅用于运行时负载，不将训练图片当成准确率验收。

分别报告：

- 模型调用阶段：Ultralytics CUDA同步inference profiler，中位数/p95。
  包含模型调用开销，排除预处理/后处理，不是纯CUDA kernel时间。
- 完整检测调用：CPU RGB输入到CPU框/类别/分数，包含格式转换、预处理、
  H2D、前向、NMS/后处理及D2H，中位数/p95。

可选增加参数：
`--episode D:\FPS_AI_Dataset\Apex_YOLO_DAY3\episode_000035`

这会额外逐模型逐组处理同一段原生时间序列，沿用threads跟踪配置，
输出检测+外观+跟踪完整视觉处理延迟；保持全部帧轨迹，排除最初30个独立观测。
这些是处理延迟，不包含视频解码、屏幕采集、视频编码和控制器输出。
模型输出数量不同会影响跟踪开销，这也是实际运行需要衡量的差异。

报告目录为 `runs/c1_detection/all_model_latency_<时间>`。
`samples.csv`含每次计时，`summary.json`含整体及每组分布，
`延迟对比报告.md`给出总表；可选跟踪结果在`visual_pipeline_blocks.json`及各模型原生逐帧日志。
不要相加各阶段p95，不以训练日志的批量验证速度当战斗单帧延迟。
15ms纯模型前向目标单独判断，capture-to-action需要另行连接实际采集和动作输出测量。

先比较PT/FP16。需要TensorRT时，另做同精度、同检测分支的导出/正确性检查后再比较，
不把某个TensorRT模型与其他PT模型混作纯模型规模比较。
