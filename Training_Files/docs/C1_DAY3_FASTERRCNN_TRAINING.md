# Day3 Faster R-CNN 敌人 / 队友训练

本实验属于 Stage 1 — Local Combat Policy 的视觉检测子任务。用户在 PyCharm 执行训练。

## 目的与数据流

同一张截图先经过 MobileNetV3-FPN 提取多尺度特征，RPN 生成候选区域，
ROI 分类头判断背景、敌人、队友，回归头调整框的位置。
模型训练同时优化 loss_objectness、loss_rpn_box_reg、loss_classifier、loss_box_reg。
它学习截图中的敌我身份；身份提示缺失时，不能通过换架构创造缺失的信息。

直接读取 `c1_yolo_dataset/day1_day2_v4_real_match`：830张训练、62张验证，
训练537个敌人框、146个队友框、311张负样本；验证34个敌人框、13个队友框。
不复制或修改原图片和标签。YOLO 0=enemy、1=teammate，在内存中映射为
Torchvision 0=background、1=enemy、2=teammate。归一化xywh转成原图像素xyxy。
随后模型内部同步缩放图像及框，保持完整画面，长边1280、短边720，内部补齐到32的倍数。

采用COCO预训练 MobileNetV3-Large-FPN；两个前景类别都从COCO person分类器及
框回归参数初始化。沿用旧基线的小anchor设置(16,32,64,128,256)。
不继承旧单类游戏训练权重，避免把旧enemy标注中的队友背景偏差带入本轮。

## 配置与比较边界

FP32、batch=2、SGD lr=0.0005、momentum=0.9、weight_decay=0.0005；
首轮前200步从0.1倍学习率线性预热，24/32轮后学习率乘0.1；最多40轮，
验证mAP50-95连续12轮无改善后早停。梯度范数上限10，非有限损失/梯度直接停止。
Windows DataLoader workers=0。seed=20261009。每次普通运行创建独立时间戳目录。

本轮不使用随机裁剪、翻转、颜色增强。与YOLO共享数据和切分，长边采样尺度相近；
优化器、增强、训练初始化与内部padding仍不同。这是实用基线比较，不是只改变架构的严格消融。

62张验证图曾参考预测结果人工修正，仅用于本轮选checkpoint。
38张新录制holdout不参与训练、早停或学习率选择；训练结束后才用于同条件对比。
之前多次查看过的holdout属于已用评估集，最终泛化仍需新的独立对局。

## PyCharm

入口：`train_c1_fasterrcnn_day3_teammates.py`

解释器：`C:\Users\15207\Documents\ChatGPT\FPS\.venv-yolo\Scripts\python.exe`

Working directory：`C:\Users\15207\Documents\ChatGPT\FPS`

1. 第一次将 Parameters 设为 `--check-only`，确认模型、CUDA、类别、数据哈希和切分通过。
2. 清空 Parameters，Run开始训练。若 Codex 已报告检查通过，可直接执行第2步。

`--smoke-test`只执行一批双类别正样本和空标注负样本的前向、反向及推理，
不更新参数、不保存训练模型；由Codex做有界接口诊断即可。

正常训练每50批报告loss，每轮结束报告验证mAP和两类P/R。
显示运行目录 `runs/c1_detection/day3_fasterrcnn_enemy_teammate_时间戳`。
每轮保留 best.pt、last.pt、history.json、results.csv、results.png、run_config.json；
训练完成另保存summary.json。

只有需要续训时才设置 `--resume "完整运行目录\last.pt"`。
恢复模型、优化器、调度器、历史和随机状态；拒绝数据/配置版本改变。
last.pt在每轮结束保存；中断后从上一个完整轮次恢复。普通Run不会自动覆盖或续训。

## 如何解释结果

- 四项loss都是训练损失；下降说明拟合训练样本，不代表新对局更准确。
- best.pt由验证mAP50-95选取，而非最后一轮或训练loss最低轮。
- mAP使用pycocotools COCOeval，IoU 0.50到0.95、101点召回、maxDet=100，预测分数保留至0.001。
- P/R分别报告固定0.05和0.25阈值，不能直接把这些工作点与AP混为一谈。
- person_recall忽略敌我类别做匹配；identity_accuracy_on_located只统计已匹配人物。
- 未匹配预测含背景误检、重复框和定位不合格，不能全部叫背景误检。
- 队友验证真值仅13个框，指标波动较大。等待独立holdout结果再判断优劣。
- loss变NaN或梯度非有限时脚本停止。提供报错及failure.json（损失异常时生成），不要直接增加轮数。
- 本训练入口不做延迟测速，完成后另用统一前处理/后处理口径测中位数及p95。

模型及target接口参考：
[Torchvision模型文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html)、
[检测微调教程](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)。

## 训练完成后的三模型对照

当前训练已在38轮早停，最佳轮次26。最佳验证mAP50=0.4806969、
mAP50-95=0.2033130；最后一轮的指标不能替代best.pt指标。

在同一PyCharm解释器与工作目录中，运行 `compare_c1_day3_fasterrcnn_yolo.py`，
Parameters留空。默认加载20261009_160808_243868运行的best.pt，
同时重新评估YOLO Day1 V3、Day2 V4。需要替换Faster R-CNN权重时使用
`--fasterrcnn-weights "完整路径\best.pt"`。

38张holdout上的所有模型统一采用pycocotools COCOeval及相同固定阈值匹配，
报告0.05/0.25两个工作点；AP分数可能与之前Ultralytics报告略有不同。
两个工作点仅用于描述，不在该holdout上自动搜索最佳阈值。
默认可视化阈值0.05；可用 `--display-threshold 0.25` 更改展示阈值，
汇总的两个工作点不会改变。原编号保留，输出高清三列逐图对照。

测速预热30次、测200次、batch=1、conf=0.05，以相同的已解码CPU RGB数组
作为输入，以CPU原图坐标框/类别/分数作为输出。包括格式转换、归一化、缩放、
padding、主机到GPU传输、模型、候选框/NMS、GPU到CPU回传；不包括解码、
抓屏、渲染、跟踪与动作输出。Faster R-CNN使用训练验证时的FP32，
YOLO使用当前部署的FP16，反映这两种配置的实际取舍。
这不是纯模型前向计时，不直接用15ms前向预算判定通过/失败。

输出目录为 `runs/c1_detection/day3_holdout_comparison_时间戳`。
包含comparison.json、中文对比报告.md、三模型对照图片，
以及每个模型的summary.json、prediction_details.json、image_review.csv和rendered图片。
脚本的 `--check-only` 只核验数据、哈希、类别和权重加载，不运行验收或测速。
