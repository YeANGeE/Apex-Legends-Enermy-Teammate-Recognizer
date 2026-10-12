# 27张外部截图：标注与三模型诊断

本流程属于 Stage 1 — Local Combat Policy。目的为识别V4的漏定位、
背景/重复预测、框偏差和敌我分类问题，不训练模型，也不评估跟踪连续性。
网络截图可能来自相同视频，不能把27张当作27个独立场景。

## 第一步：人工标注

独立项目：`外部截图三模型诊断 V1 — 27张`。
访问 `http://127.0.0.1:8080/projects/14/data`。
如服务未启动，在PyCharm运行现有 `run_c1_label_studio.py`；不需要参数。
沿用现有启动解释器；该脚本调用 `.venv-label313/Scripts/label-studio.exe`。
工作目录是FPS项目根目录。运行后应出现8080地址。

- 标每个可确认敌人和队友的可见身体，不包含头顶ID、血条或字幕。
- 身体被遮挡/截断时框可见范围，不猜测图外或完全遮挡的身体。
- 第一人称自己的手臂与武器、头像、纯隔墙标记、播放控件不属于目标。
- 小人物不要省略；没有ID不能直接判敌人。
- 任何人物身份或范围无法可靠确认时整图选“不确定”；代码排除该图。
- 清晰且没有人物的图选“可用”，提交空框；这才是有效的负例。
- 场景标签可多选，推荐完整填写。原图含字幕/播放控件可选相应标签。

在Label Studio逐图点Submit；只保存草稿不算完成。
全部完成后，Export选择完整 **JSON**，不选JSON-MIN、YOLO或COCO。
将导出文件保存为项目下：
`c1_annotations/external_screenshots_v1/annotations.json`。
每图保留一份最终人工标注；缺图、多个标注、取消标注都会明确报错。

## 第二步：PyCharm对比

原理：三个模型读取完全相同原图，以各自现有长边1280配置预测。
人工框统一转换为原图像素坐标，统一用COCOeval计算AP，
在conf=0.05和0.25、IoU≥0.5下计算召回/精确率和身份混淆。
Faster R-CNN采用FP32，YOLO采用FP16；是现有配置的实用对比。

PyCharm配置：

- 脚本：`compare_c1_external_screenshots.py`
- 解释器：`C:\Users\15207\Documents\ChatGPT\FPS\.venv-yolo\Scripts\python.exe`
- 工作目录：`C:\Users\15207\Documents\ChatGPT\FPS`
- 第一次参数：`--check-only`。预期显示可用/不确定/人物数量与检查通过。
- 检查通过后清空参数，再运行正式对比。不会训练或输出控制器动作。

默认权重：

- YOLO V3：`runs/c1_yolo/day1_yolo26n_1280_v3_enemy_teammate/weights/best.pt`
- YOLO V4：`runs/c1_yolo/day2_yolo26n_1280_v4_real_match-2/weights/best.pt`
- Faster R-CNN：`runs/c1_detection/day3_fasterrcnn_enemy_teammate_20261009_160808_243868/best.pt`

结果目录：`runs/c1_detection/external_screenshots_<时间>/`。
代码审计图片与既有数据集的完全文件/像素重复；这不能识别相邻帧和近似截图。
run_config记录导出、图片清单与模型文件哈希，human_truth保存实际评估标注。

## 如何解释输出

- `对比报告.md`：同一批图的总表、V4失败图片编号、尺寸分组计数。
- `compare_*.png`：三模型原尺寸无损并排拼接，顶部增加96像素标题栏；以100%缩放查看细节。
- `object_diagnosis.csv`：每个人工目标是否正确、认反或漏定位/框偏差。
- 各模型`image_review.csv`：逐图检查多框、漏框和身份错判；其绘图匹配沿用类别优先规则。
- 各模型`scene_metrics.json`：按人工场景标签统计，报告图片数与人物数。
- `size_metrics.json`：按身体高度分组，不能把“小框”等同于远距离。
- 各模型`prediction_details.json`：保留低置信度预测，方便后续复核。

汇总身份诊断与逐目标诊断使用类别无关匹配，先问“找到人了吗”，
再问“身份对了吗”。已定位身份准确率不能代表全部人物准确率。
未匹配预测可能是背景、重复框或定位失败，必须结合图复核。
V4补强建议是待核查方向，最终补什么数据需依据这些可见错误决定。

可选参数`--benchmark`会额外预热30次、测速200次。
该计时从已解码CPU RGB到CPU预测框，包含预处理和后处理，
不包含截图采集、追踪、绘图或控制器输出，不等同于纯模型前向或capture-to-action。
当前先完成准确性诊断即可。

这批图用于改进V4后只能作为诊断集；若以后选择其中图片训练，
需要另找独立来源的测试集评价改进，不能继续用同批图证明泛化提升。
