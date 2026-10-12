# Stage 1只读实时叠加测试

PyCharm入口preview_c1_live_overlay.py，.venv-yolo/Scripts/python.exe，工作目录FPS，参数留空。
用户启动，Codex不会后台启动采集服务。默认180秒，F8隐藏/显示，F10退出；也可按PyCharm停止。
游戏应使用无边框窗口或窗口模式，初次在训练场测试。覆盖DXGI device0/output0整个显示器，框不是绘制进游戏进程。
如果采集了另一台显示器，指定--output-index 1；必要时结合--device-index选择DXGI适配器。
窗口位置读取同一DXGI输出的DesktopCoordinates；全屏图缩放到1920x1080推理，框再按显示器比例映射回去。
显示模式或分辨率变化后退出再启动；首版不支持旋转显示器。

固定主候选：DAY4补样best，SHA a2ceeb4de51d51d9a422014de3a0a657aeba5c7f645e09188a584227721a93c7，阈值0.10；advanced/threads，面积修正开启，重复框分组关闭。
红敌人、蓝队友、灰不确定、绿当前选择。只观察DXGI画面与退出/隐藏键，不注入任何鼠标、键盘或手柄动作。
透明置顶层鼠标穿透、禁止激活；WDA_EXCLUDEFROMCAPTURE设置及读取失败则停止，避免将框再送入检测。
官方依据：https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwindowdisplayaffinity
接口通过并不等于游戏模式下可见效果通过。首次运行需确认覆盖对齐、输入穿透、截图中不含本程序绘制的框。

捕获回调只覆盖一个最新帧槽，推理不积累待处理队列。DXcam私有环读取依赖已核验0.3.0版本。
源时间为DXGI LastPresentTime→QPC，重复时间不重复更新轨迹；源帧已过150ms则跳过，显示中的框超过150ms则清空。
图像处理在独立工作线程，Tk主线程提交框；不绘制预测框、不编码视频。
输出runs/c1_live_overlay/<时间>/run_config.json、frames.jsonl、summary.json，以及第60个处理帧的采集原图PNG。
原图应用来检查是否有自己的叠加框被采入；不包含叠加框时不能拿来直接判断框绘制效果。

处理指标跳过初始30个已处理帧：predict调用含预处理/传输/后处理，appearance、tracking（含选择）、processing（含CPU缩放）分别记录。
source_age_at_finish为源画面到处理完成，source_age_at_ui_submit为源画面到Tk软件绘制提交；不包含真正屏幕扫描显示时间，不是捕获到动作延迟。
latest_slot_overwrites属于主动丢旧帧策略，不能单独解释为失败；结合画面年龄与采集错误看。
真正验收需用户确认：静态对齐、转视角时框滞后、遮挡后编号、队友误锁及场景误检。先取得真实运行结果再讨论优化。

## 首轮不可见排查

用户在无边框游戏中报告叠加不可见；结果20261010_195904_230084。
1470处理帧中599帧有可绘制轨迹，软件UI提交年龄p95=110.46ms，处理p95=82.75ms；不能用无检测或全部过期解释不可见。
捕获经历显示状态变更并成功恢复，捕获错误计数0；提示不是致命错误，但需结合可见效果确认。
修正候选revision2：Tk映射后重设原生穿透/不激活/分层样式并核验采集排除；每500ms使用SetWindowPos(HWND_TOPMOST, SWP_NOACTIVATE)恢复置顶。
左上常驻绿色诊断标记不随过期框清除，F8可隐藏整层。启动信息按实际seconds打印。
此修正通过静态依赖/测试检查，未由Codex启动窗口；不可见问题尚待用户验证。
复测参数--seconds 60：先在桌面确认绿色诊断标记，再切入已确认的无边框游戏，确认标记是否仍存在。若仅游戏中消失，继续调查显示合成/窗口层级；若桌面也没有，调查叠加窗口创建/几何。
官方接口：https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwindowpos

## 实时可见后的滞后

用户确认叠加可见，但滞后超过一个身位，并有跳号、场景误检、敌我认反；实时验收失败。
结果20261010_200140_775931：predict中位35.68ms/p95 53.41ms，appearance中位8.67ms/p95 22.94ms，processing中位59.27ms/p95 85.19ms。
源画面到UI软件提交中位86.92ms/p95 113.09ms；处理观测间隔中位66.38ms/p95 83.64ms，不等于60Hz连续感知。
逐帧processing减去predict/appearance/tracking后的剩余中位8.37ms，主要包含4K缩放与颜色复制。
不采用位置外推伪装新观测，不把认反/场景误检解释为纯延迟；候选主模型的离线改进不代表实时可用。
颜色转换改为cv2.cvtColor，保持完全相同RGB像素，测试验证。
增加可选--tracker motion：仅关闭外观编码，保留相机/面积修正、检测模型、0.10阈值与身份规则；默认仍advanced。
用户在相同游戏场景用--tracker motion --seconds 60对比框滞后；位置关联可能造成更多换号/误关联，不能直接晋升。
下一阶段若检测调用仍是瓶颈，应评估当前权重的TensorRT同分支导出及精度一致性，不能直接套用旧权重引擎。

## TensorRT低延迟候选

当前补样主模型已导出至runs/c1_tensorrt/20261010_201904_287394/detector.engine，FP16、1280、batch1、dynamic=False、nms=None、max_det100。
导出回执核验原始PT哈希及engine哈希，原始PT未改；构建成功不是速度或精度验收。
转换过程中导出器安装了onnxruntime-gpu1.31.0及flatbuffers25.12.19作为ONNX简化依赖；未升级Torch/Ultralytics。
benchmark_c1_day4_engine.py：用户PyCharm运行，无参数；对同60张补样图片逐对交替执行PT/TRT，预热后报告predict wall time和框/类别一致性。
补样图片用于转换一致性检查，不用来重新宣称模型泛化准确率。不得将predict wall time称为纯forward或实时capture-to-action。
preview_c1_live_low_latency.py：用户在转换比较通过后运行，参数--seconds 240；固定TRT/motion/120Hz采集/2msUI轮询，其他检测条件保持。
没有启动实时采集或用户性能测试；11项相关代码测试、两入口check-only通过。
默认preview_c1_live_overlay.py仍为PT/advanced/60Hz/8ms，便于回退；低延迟候选尚未提升为默认。
候选需确认真实游戏中的source_age_at_ui_submit median/p95、框滞后、编号变化和敌我表现。
120Hz是请求采集频率，实际来源新帧频率需看capture_stats；重复DXGI时间戳不会重复推理。

## 持续优化V2

用户授权自动继续准备优化，直到要求停止或报告效果变差；实时测试与启动仍由用户在PyCharm执行。
基线为低延迟V1结果20261010_202504_902167：画面到UI提交median42.81ms/p95 57.24ms。
preview_c1_live_low_latency_v2.py相对V1仅增加latest_only采集：采集环有多帧待读时直接读取最新位置；相同源时间戳免去4K图像复制。
相机参数、TensorRT模型、检测阈值0.10、1280输入、motion跟踪、120Hz请求和2msUI轮询保持。
DXcamCapture默认latest_only=False，录像服务继续顺序读取；V2不是录制入口，不对录制主动丢帧。
新增capture_stats.realtime_skipped_frames与duplicate_copies_avoided，分别报告主动丢旧帧和省略重复拷贝；不伪装物理环覆盖。
7项相关测试及入口check-only通过，尚未实测改善。
用户运行V2参数--seconds 300后发送结果及同场景体感；若效果变差，回到preview_c1_live_low_latency.py。
后续候选才考虑降低镜头补偿点数/分辨率，不能在本轮同时改变，避免无法归因。

## V2反馈及V3

V2结果20261010_203502_216894，用户认为有改善，且游戏60FPS比300FPS框滞后更少。
processing median16.33ms/p95 23.21ms，source_age_at_ui_submit median40.18ms/p95 52.13ms。
相对V1软件提交年龄median42.81→40.18、p9557.24→52.13；用户试过不同帧率，不能将差异全部归因V2代码。
realtime_skipped_frames=0，duplicate_copies_avoided=4272；本轮没有出现采集环积压，因此主要启用的是重复拷贝规避，不能宣称最新环跳帧减少了实际积压。
高游戏帧率造成GPU/CPU及桌面合成争用是推测，日志没有游戏FPS/GPU分段遥测，尚不能确认为因果。
V3相对V2只启用CAMERA_PROFILES.fast：镜头补偿宽640→480、特征点350→200，线程仍4；检测1280/0.10保持。
可能加快镜头补偿，但也可能降低快速转视角、遮挡下的关联质量；未经用户实测不得晋升。
对照：在游戏固定60FPS的同一训练场分别运行V2、V3，每轮参数--seconds 120 --game-fps-label 60。
game-fps-label仅记录用户设定，不自动调整游戏、不声称实际FPS是60。比较画面年龄、跟踪耗时和同人换号；若劣化回退V2。

## V3实测与V4

V2结果20261010_204624_032255：tracking median5.93/p957.52ms，processing median14.33/p9517.41ms，UI提交年龄median36.50/p9549.58ms。
V3结果20261010_204823_183958：tracking median3.67/p954.81ms，processing median11.60/p9515.02ms，UI提交年龄median34.42/p9544.59ms。
用户确认两轮游戏均限制60FPS，V3延迟稍好，其他无明显差别；据此保留V3为当前优化参考，但不同实时场景不等于严格同帧对比。
下一候选preview_c1_live_low_latency_v4.py只增加进程1ms timeBeginPeriod请求，正常退出/异常清理timeEndPeriod配对；模型与跟踪参数保持V3。
同时增加ui_poll_interval记录真实UI回调间隔；原先after(2)只代表请求，不证明实际2ms调度。
官方：https://learn.microsoft.com/en-us/windows/win32/api/timeapi/nf-timeapi-timebeginperiod
V4九项相关测试与check-only通过；未启动游戏预览，调度收益尚待实测。若CPU调度成本导致劣化，回退V3。
继续固定游戏60FPS，PyCharm V4参数--seconds 300 --game-fps-label 60。

## V4通过与V5

用户确认V4静止视角人物追踪更快，转动视角滞后仍大，与V3相比无明显劣化，总体提升。
结果20261010_205505_042878：processing median11.42/p9514.41ms，UI提交年龄median29.15/p9537.31ms；真实UI回调间隔median2.24/p955.61ms。
保留V4为当前优化参考，既有镜头转动现象不能单凭本轮日志解释为转动时耗时增加。
V5相对V4只启用--event-wakeup：新帧回调设置事件，推理线程被唤醒后取得最新槽内容；默认旧入口仍2ms轮询。
事件在同一锁内设置/清除，不累积队列，关闭时等待最多20ms；不会外推旧框假装新观测。
新增每帧camera_status/camera_matrix日志供区分镜头运动与时间开销；仅诊断字段。
10项相关测试及check-only通过。PyCharm运行preview_c1_live_low_latency_v5.py，参数--seconds 300 --game-fps-label 60，游戏仍固定60FPS。
若效果变差回退preview_c1_live_low_latency_v4.py；V5尚待实测，不承诺大幅收益。

## V5无实质延迟收益

结果20261010_205920_130468，用户认为肉眼改善不大。
V4→V5：处理median11.4167→11.68355ms，p9514.41325→14.64523ms；源画面到处理完成median27.6004→27.58125ms，p9535.56135→35.702805ms；到UI提交median29.14645→29.1739ms，p9537.31471→37.5888ms。
仅开始处理前画面年龄median16.1716→15.9214ms下降0.25ms，而下游年龄无实质下降；不同场景与不同测试时长，不能宣称显著变差或提升。
V5不晋升，当前有效参考继续V4。下一步聚焦源画面到采集回调/推理开始的约16ms，先拆分该阶段，避免继续盲调轮询。
控制以后应直接读取处理完成结果，而不经过UI；处理完成年龄约27.6ms才是现阶段更接近控制输入的新鲜度指标，不是实际控制/射击延迟。

## 采集阶段拆分

diagnose_c1_live_capture_latency.py保持已验收V4，不启用V5事件唤醒；增加可选capture-timing。
FramePacket新增可选copy_start/copy_end/ready计时，默认None；录像对齐的timestamp_ns仍为DXGI LastPresentTime，默认录制不启用诊断。
source_to_capture_copy：源呈现时间到开始复制DXcam环中的图像；包含捕获后端/生产者处理及环读取调度，不能独立解释为GPU读回耗时。
capture_copy：环图像到独立NumPy图像的复制。
capture_dispatch：复制完成到准备交付包。
capture_to_inference：包准备交付到推理线程开始处理，包括回调/槽等待/调度。
同帧四项之和等于source_age_at_start；分项median/p95不能相加代替总体分位数。按QPC原生坐标验证时序，不使用视频帧号或未来动作。
14项相关测试及check-only通过，未启动实时诊断。用户固定游戏60FPS，参数--seconds 120 --game-fps-label 60；发送结果目录。
诊断入口只找瓶颈，不宣称有新优化收益。下一步按实际占比选择后端读取或线程交付优化。

## 采集诊断结果与V6

结果20261010_211043_770275：source_to_capture_copy median7.95/p9512.34ms，capture_copy median2.83/p953.84ms，capture_dispatch median0.008/p950.016ms，capture_to_inference median4.32/p959.13ms。
源画面到开始处理median15.69ms，源画面到UI提交median28.67/p9536.39ms。
上述分项不能按各自median/p95相加；回调到推理等待包括上一帧计算未结束及调度，不等于单纯睡眠。
V6相对已验收V4（与诊断入口相同配置）只改交付图像路径：在DXcam环锁内用INTER_AREA直接生成1920x1080独立图像，不再先拷贝3840x2160然后缩放。
锁内操作防止生产者修改源图，输出不共享环内存；测试确认直接缩放像素与复制后缩放一致。原尺寸与坐标映射仍来自显示器，检测输入参数不变。
录制路径默认不启用downscale_for_preview，保持原行为；此功能只允许实时latest_only入口。
V6的capture_copy计时包含缩放，prepare阶段不再重复缩放；整体source_age_at_finish/UI age用于最终比较，避免只比较移动后的分项。
16项相关测试及check-only通过，实际收益未验证。PyCharm运行preview_c1_live_low_latency_v6.py，参数--seconds 120 --game-fps-label 60。
保持游戏60FPS；若滞后、流畅度或识别变差，回退V4，诊断数据保留用于下一轮。

## V6通过与V7颜色路径候选

结果20261010_211515_802106，用户明确确认没有变差并继续优化；V6保留为当前参考。
相对V4采集诊断：processing median11.33→9.48ms，处理完成年龄median27.13→21.63ms，UI提交年龄median28.67→22.93ms、p9536.39→30.13ms。
capture_copy median2.83→0.95ms（V6包含缩放），prepare median2.06→0.66ms；source_to_capture_copy仍median7.98ms，下一瓶颈在采集后端前段。
V7只改变采集颜色路径：DXcam环BGRA，缩到1080p后转BGR；原来在4K时转BGR。
基于本地DXcam 0.3.0处理器源代码；BGRA路径仍有数据读取/复制，不宣称零拷贝。4通道增加带宽，可能抵消转换收益，必须实测。
使用3840x2160随机BGRA输入验证缩放后转BGR与先转BGR再缩放完全相同，输出为独立1920x1080x3图像。
17项相关测试及check-only通过；默认录制/BGR路径保持。用户运行preview_c1_live_low_latency_v7.py，参数--seconds 120 --game-fps-label 60。
如延迟、流畅度或识别劣化，回退V6；不改变模型、阈值、镜头补偿或身份确认规则。

## V7未取得总延迟收益，保留V6

结果20261010_211947_609653：UI提交年龄median24.12/p9530.61ms，处理完成年龄median22.74/p9529.12ms；V6对应22.93/30.13与21.63/28.60ms。
source_to_capture_copy median7.98→6.93ms，但capture_copy（包括缩放及丢alpha）median0.95→1.93ms；没有明显净收益。
predict/跟踪也略变慢，实时场景不同，不能将所有波动归因BGRA或宣称统计显著劣化。
V7不晋升，BGRA颜色路径优化停止，使用preview_c1_live_low_latency_v6.py作为当前参考。保持V7日志与入口作历史对照，不修改旧结果。
用户允许效果变差时回退；当前更保守选择是V6，不继续叠加未经证实的BGRA改动。
