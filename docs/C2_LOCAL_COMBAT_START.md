# Stage 1 — Local Combat Policy：C2起步

本项目已经在Stage 1，进入C2是同一阶段内从视觉感知走向动作学习。架构基线未正式定义C2编号；本文按本轮讨论将其作为局部瞄准/动作学习子任务工作名，不代替ARCHITECTURE_AND_ROADMAP.md。

## C1工作基线

源模型day4_yolo26n_1280_joint_supplement_v1/weights/best.pt，SHA a2ceeb4de51d51d9a422014de3a0a657aeba5c7f645e09188a584227721a93c7。
TensorRT engine runs/c1_tensorrt/20261010_201904_287394/detector.engine，SHA de2612173c611a428ff1ee89faef382950150d9df15b095db895492dd6547f1f。
实时路径preview_c1_live_low_latency_v6.py：1280检测、conf0.10、motion、fast镜头补偿、面积修正、重复分组关闭、120Hz请求采集、直接交付1080p、1ms定时请求。
V6在游戏60FPS下source_age_at_finish median21.63/p9528.60ms，UI提交median22.93/p9530.13ms。无控制输出，不是捕获到动作延迟。
V7未获净延迟收益，保持历史候选，不用于C2默认。
基线支持研究，不证明队友误识别、背景误检和编号跳变已解决；C2不能把这些观测当作无误真值。

## 第一条学习链路

按架构基线使用模仿学习/行为克隆，首先建立observation_t→RX/RY/RB/LB数据与小模型。
先做离线单片段拟合验证整条训练链路，重点看RX/RY，RB开火/LB瞄准保留为联合输出；随后扩大数据，在保留录像验证。
输入包括图像及C1当前检测/目标几何、有效性和观测时间；不能只使用人物框中心，否则无法充分表达准心、开镜、武器、遮挡等信息。
实时C2直接读取带源时间戳的感知结果，不经过绘图；观测过期/类别冲突/目标丢失应明确标记。
先进行影子模式只显示预测手柄动作，不输出控制；再测完整感知+动作预测链路。后续闭环操作需要独立实现和验收。
单段拟合只验证程序可学习，不证明会跟枪、开火安全或泛化。

## 首轮机械审计

已检查DAY4 E2/E4/E10/E11/E12的frames.csv与xinput.csv，共1667帧；每帧均有不晚于source_present_ns的输入记录。
明确采用xinput.csv，不与gamepad.csv混用；按timestamp_ns<=source_present_ns最新记录做因果向后对齐，不按行号、不取未来动作。
各片段对齐记录年龄p95约0.24–0.25ms，只是主机记录采样新鲜度，不能解释为控制器硬件延迟或人体反应延迟。
审计diagnostics/c2_initial_alignment/audit.json；没有训练、模型验证或手柄输出。
这五段仅作为首轮数据检查，开火覆盖不足（E2无RT>0.1，其余输入行约1.4%–4.9%），不能直接作为完整开火学习数据。
先从现有DAY3/DAY4录像筛选完整持续交战片段，再决定是否需要补录。
DAY4 E1/E3/E6/E7/E9继续保留验证，禁止把它们自动混入C2训练；不同episode不能证明来自独立比赛。

## DAY1–4首轮代码与输入来源

用户要求使用现有各DAY数据训练；离线APEX每轮60秒，用户表示设置理论上一致，以命中率报告。仍需实际确认输入映射及闭环响应，不能仅凭理论一致视为已校准。
本轮为跨DAY共同动作来源显式选择gamepad.csv（GameInput），与先前五段XInput可行性审计不同；不静默混用，不改变原始记录。
prepare_c2_recording_manifest.py完成784段初步元数据/动作检查：DAY1 198、DAY2 215、DAY3 320、DAY4 13、原DAY2_VAL 38。
分配694训练片段76166采样点，20个DAY1已有交战分组验证片段2246点，20个DAY1原测试片段2181点，50个保留诊断片段5413点。
DAY1沿用c1_day1_formal_manifest.csv的交战组划分；DAY2-4比赛边界未知全部候选训练，原DAY2_VAL及DAY3七段/DAY4五段不入训练。该验证不充分代表全部DAY泛化；也未证明不同目录一定不同比赛。
原Apex_Day2两段未自动并入，因来源角色不同，先保留而非猜测。
每隔至少50ms源时间选择非重复帧，动作按最新GameInput timestamp_ns<=source_present_ns，超过100ms无更新的标签排除。

冻结C1引擎只提供观测：top8检测框位置/尺寸、置信度、敌我假设；它不提供动作真值，当前未把外观ID当跨视频人物标签。
C2输入3张160x288画面（按原生时刻取约t-100ms、t-50ms、t的因果历史）、当前C1 56维检测特征及真实历史间隔。
输出RX/RY经tanh、RB/LB按下概率经sigmoid；SmoothL1模仿人工动作。不假定只凭框中心就能学习压枪/开镜。
prepare_c2_policy_cache.py默认准备DAY3 episode265单片段；已完整解码，生成75个对齐观测。C1推理使用1280，不因C2小图降低检测分辨率。
缓存使用独立.npy及mmap，避免将约7.6万个RGB样本一次全部复制到内存。缓存收据固定源记录/视频、C1引擎及数组哈希。
4项因果对齐/历史/类别特征/模型范围和反传测试通过；train_c2_behavior_cloning.py --check-only通过，73个单段历史样本，未启动训练。

当前PyCharm操作：运行train_c2_behavior_cloning.py，.venv-yolo解释器，工作目录FPS，参数留空；默认单段150轮，学习率0.001、batch32。
输出runs/c2_bc/overfit_<时间>/best.pt、last.pt、results.json；action_mae顺序RX/RY/RB/LB，并提供训练均值常量预测基线。单段只做程序拟合检查，无独立验证或游戏输出。
单段检查通过后，用户运行prepare_c2_policy_cache.py --all完成全部train/val完整解码及C1缓存，再运行train_c2_behavior_cloning.py --all联合30轮。不要提前跳过单段结果。
完整训练best.pt按四动作MAE均值选择；不是射击命中率最优保证，不能因为少开火误差低就认定会交战。
闭环前先影子输出，直接读V6感知结果而非绘图；目标/观测有效性与动作约束需另做集成验收。
靶场每轮固定60秒、武器、距离/目标模式及设置；同时记录命中率与射击量/命中量，防止“少开火”伪提高。
旧训练录像用于模仿学习，靶场命中率目前是评估信号，不会自动把单个总分当作每帧监督标签；若以后做奖励学习，需可追踪的动作/奖励接口。

## 用户键位修正（当前有效）

完整键位见docs/APEX_CONTROLLER_BINDINGS.md及apex_controller_bindings.json。LB按住瞄准、RB攻击、RT跳跃、LT按住蹲伏。
原day1234_bc_v1及DAY3 episode265的LT/RT标签结果为历史版本，不能用于当前开火/瞄准。
新目录c2_dataset/day1234_bc_rb_lb_v2，按原始rb/lb字段重建全部694训练候选及既有留出数据的标签，不重命名旧扳机数值。
重新选DAY2 episode15，完整解码和C1特征准备223观测，221历史训练样本；按钮标签包含按下与释放。
6项测试及训练check-only通过；保持用户已有旧结果不删除，没有启动新训练。
预处理出现一次NMS耗时警告；核对本地nms.py，该警告在output[xi]赋值后触发。逐图batch1因此保留已完成的该图输出，没有未处理的后续批图；不将此冷启动耗时当稳定延迟。
