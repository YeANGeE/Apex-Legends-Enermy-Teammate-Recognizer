"""User-run early-stopping ablation: same generic YOLO26n recipe, full 60-epoch budget."""
import train_c1_yolo_joint_pretrained as training


if __name__=="__main__":
    training.SETTINGS={**training.SETTINGS,"epochs":60,"patience":0,
                       "name":"day3_yolo26n_1280_joint_pretrained_full60_v1"}
    print("训练预算对照：通用YOLO26n，同一数据和60轮学习率计划，仅关闭早停(patience=0)。",flush=True)
    print("新实验重新初始化，不修改当前正在运行的n；best.pt仍按旧验证集最佳指标保存。",flush=True)
    training.main()
