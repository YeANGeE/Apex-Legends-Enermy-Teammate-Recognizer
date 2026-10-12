"""User-run YOLO26s scale comparison: same merged data and recipe as generic YOLO26n."""
from pathlib import Path
import train_c1_yolo_joint_pretrained as training

ROOT=Path(__file__).resolve().parent


if __name__=="__main__":
    training.WEIGHTS=ROOT/"weights/yolo26s.pt"
    training.EXPECTED_SHA="646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
    training.SETTINGS={**training.SETTINGS,"name":"day3_yolo26s_1280_joint_pretrained_v1"}
    print("YOLO26s模型规模对照：与原始26n联合训练保持相同数据、尺寸、batch、学习率、增强和训练预算。",flush=True)
    print("batch=4为初始设置；若OOM应保留报错再调整，不能静默改变对照条件。",flush=True)
    training.main()
