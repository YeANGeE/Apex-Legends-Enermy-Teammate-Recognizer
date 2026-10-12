# C2联合训练后的离线动作诊断

当前候选为runs/c2_bc/joint_20261010_222327_686748/best.pt，第27轮。
30轮已完成，验证MAE RX/RY/RB/LB为0.21364/0.17641/0.13983/0.07214。
该权重是GameInput监督基线，不包含高频XInput惯性学习，不代表闭环命中率。

用户在PyCharm运行evaluate_c2_joint_offline.py，解释器.venv-yolo，工作目录FPS，参数留空。
脚本默认只读已有20段DAY1验证缓存，核对权重架构、动作映射、manifest和缓存哈希，
不读取新训练录像、不重新训练、不发送手柄信号。check-only已通过，7项相关测试通过；
完整模型诊断由用户执行。

输出runs/c2_offline_review/<时间>/results.json及predictions.csv。
MAE为动作数值误差；abs_error_p95观察较大偏差；摇杆方向统计仅在人类绝对输入至少0.1时计算。
按钮按0.5固定诊断阈值报告precision/recall、误按/漏按和按下比例；该阈值不是实机验收参数，
sigmoid数值也未证明经过概率校准。指标为空表示无对应分母，不视为满分。

这批验证已用于选best.pt，不能称为未接触的最终测试；不能由其推断DAY2-4或靶场泛化。
原测试/诊断片段仍保留。predictions.csv按原样本source_present_ns导出，不按行号对齐控制输入。
下一步依据诊断定位轴方向与按钮问题，再准备只读影子预览；闭环和1000Hz惯性输出另行验收。
