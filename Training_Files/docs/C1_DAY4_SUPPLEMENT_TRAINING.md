# DAY4补样联合训练

人工导出：c1_yolo_export/DAY4/4.1，Label Studio项目18。
E2/E4/E10/E11/E12每0.5秒抽帧，共60张，全部可用；18张负样本，39敌人框、21队友框，JSON/YOLO ZIP/原图一致，无标注格式警告。
仅采用人工结果；预标注不是训练真值。

新数据集c1_yolo_dataset/day4_joint_supplement_v1：1791训练图、62旧验证图。
新增60张，无精确重复；为避免与旧DAY3同episode文件名冲突，新图加入day4_supplement_前缀，像素及标注未改。
训练共544负样本、1262敌人框、474队友框。旧验证图和标签哈希保持。
DAY4 E1/E3/E6/E7/E9来源未进入数据集；仅按来源隔离，未证明比赛独立或无近似重复。
manifest SHA256：214e089aa1241c4da76bfddde1ac1194d3d6ead7bb82059cf5d5d34d7d0bc826。

运行train_c1_yolo_day4_joint_supplement.py，.venv-yolo/Scripts/python.exe，项目根目录，参数留空。
从通用weights/yolo26n.pt开始，联合全部1791张，1280、batch4、AdamW、lr0=0.001、最多60轮、patience20，其他参数沿用前轮。
check-only已通过，Codex未启动训练。
输出runs/c1_yolo/day4_yolo26n_1280_joint_supplement_v1，含best.pt/last.pt/results.csv/data_receipt.json。
best.pt由旧验证指标选择；最终改进仍要在保留的五段DAY4视频上人工确认场景误检、队友认反及小目标漏检，并使用一致跟踪设置比较。

训练结果核对：完成31轮；CSV最佳第11轮，AP50=0.71469，AP50-95=0.35572，P=0.72466，R=0.69177。
上一版DAY4错误补标模型CSV最佳AP50-95=0.34402，差值+0.01170；旧验证指标不等于视频泛化准确率。
best/last均epoch=-1、optimizer=None、类别enemy/teammate，浮点权重有限；已最终保存。
best SHA256：a2ceeb4de51d51d9a422014de3a0a657aeba5c7f645e09188a584227721a93c7。
运行demo_c1_day4_supplement_tracking.py，参数留空；阈值0.10、stable_geometry=True、group_duplicates=False。
对应旧模型同设置结果为runs/c1_tracking/20261010_174726_891310，不与重复框分组开启版直接归因比较。
重点位置：E1同人换号；E3帧230/280队友；E6原断轨；E7所有场景误检；E9帧24/364/413/614/642。
