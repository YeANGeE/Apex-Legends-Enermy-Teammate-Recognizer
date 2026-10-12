"""User-run Day4 best on unchanged seven-clip tracking regression."""
from pathlib import Path
import sys
import demo_c1_video_tracking as demo
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent
WEIGHTS=ROOT/'runs/c1_yolo/day4_yolo26n_1280_joint_errors_v1/weights/best.pt'
EXPECTED_SHA='034e5d999ec96ea91778854d81eec881c3da4d89f397a165d872f6e68dc19945'


if __name__=='__main__':
    if sha(WEIGHTS)!=EXPECTED_SHA:raise ValueError('Day4 candidate checkpoint changed')
    demo.WEIGHTS=WEIGHTS;demo.SOURCE=Path('D:/FPS_AI_Dataset/Apex_YOLO_DAY3')
    print('DAY4错误补标候选，第33轮best：同七段视频、同检测和跟踪设置。',flush=True)
    print('检查E43队友认反，E47枪械配件，E79倒地队友与未关联敌人，E82/E150漏检，E178字幕，E320树与准心。',flush=True)
    print('左检测、右跟踪和目标选择，原尺寸拼接；这些帧已用于训练，仅是回归诊断，不控制游戏。',flush=True)
    demo.main(['--episodes','43','47','79','82','150','178','320','--split-view','--select-target',*sys.argv[1:]])
