"""User-run YOLO26m joint training, same data and recipe as the n/s experiments."""
from pathlib import Path
import train_c1_yolo_joint_pretrained as training

ROOT=Path(__file__).resolve().parent


if __name__=="__main__":
    training.WEIGHTS=ROOT/"weights/yolo26m.pt"
    training.EXPECTED_SHA="401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"
    training.SETTINGS={**training.SETTINGS,"name":"day3_yolo26m_1280_joint_pretrained_v1"}
    print("YOLO26m规模对照：与26n/26s保持相同数据、1280输入、batch=4、学习率、增强和训练预算。",flush=True)
    print("若出现OOM，保留错误反馈；不静默改变batch或分辨率。",flush=True)
    training.main()
