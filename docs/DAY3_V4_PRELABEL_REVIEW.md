# Day3录制素材：V4预标注与人工复核

阶段：Stage 1 — Local Combat Policy，敌人/队友视觉检测。
原始素材：`D:\FPS_AI_Dataset\Apex_YOLO_DAY3`，320个episode。

准备代码完整解码每段，检查帧数、原生时间戳、录制状态和输入记录。
按RB触发时间，每段最多选两张非重复帧，时间间隔至少0.75秒；
帧序号仅用于取出已经按source_present_ns选定的图像，不用于动作标签配对。
所有图片保持1920×1080原尺寸，以PNG保存。去掉完全像素重复图片；
相似但并不完全相同的图片保留人工判断，不自动删除小人物变化。
录制警告写入来源清单，输入采样警告不自动等同于图像不可用。

原图、旧标注和模型保持不变。V4以1280输入、conf=0.05、FP16预标。
含零预测的图片照样导入；预测没有被写为人工annotations。
当前split=unassigned、match_group=unconfirmed，不宣称封存独立测试。
多局比赛的边界用户已不记得；后续需要来源组确认或另录独立测试素材。

## 用户操作

在既有Label Studio打开项目 **DAY3 V4预标注复核 V1**。
如果服务未运行，在PyCharm运行`run_c1_label_studio.py`，不需参数，
工作目录为FPS根目录。该脚本调用独立Label Studio环境。
Codex不会启动或停止该服务。

每图检查：

1. 补所有漏掉的敌人和队友，小目标不能省略；修正认反。
2. 每人一个可见身体框，不包括头顶ID、血条、头像、自己的手臂武器。
3. 框只是松紧不同，按同一规则统一，不称为身份失败。
4. 预标为空：确实无人选“可用”提交空框；有人则补框；判断不了选“不确定”。
5. 菜单、结算或不保留的图选“不保留”；不用删除文件。
6. 点击Submit提交，草稿不算完成；场景标签可选，作为后续统计辅助。

全部完成后Export选择完整JSON，保存到新建目录
`c1_yolo_export/DAY3/day3_v4_prelabel_v1`，告诉Codex路径。
Codex自动校验导出、整理有效标签和统计类别/难例，再安排下一步。

## 可重用机械入口

- `.venv-yolo/Scripts/python.exe prepare_c1_day3_recordings.py`：准备与预标。
- `--prelabel-only`：仅基于已经冻结的图片清单重新预标，避免重新解码。
- `.venv-label313/Scripts/python.exe import_c1_day3_annotation.py`：有界导入，带数据库备份；不启动服务。

本步不是训练或性能评估。用户后续仍在PyCharm执行实质训练和验收。
