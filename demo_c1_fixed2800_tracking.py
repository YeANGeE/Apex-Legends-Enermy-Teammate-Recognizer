"""User-run candidate on the same seven DAY3 diagnostic clips, same tracker."""
from pathlib import Path
import sys
import demo_c1_video_tracking as demo
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent
WEIGHTS=ROOT/'runs/c1_yolo/day3_yolo26n_1280_joint_fixed2800_v1/weights/best.pt'
EXPECTED_SHA='76cc615ca2c6ced6566fd42cd137ef8744478fb8a9886dd1b9c65459327ace9c'


if __name__=='__main__':
    if sha(WEIGHTS)!=EXPECTED_SHA:raise ValueError('Candidate checkpoint changed')
    demo.WEIGHTS=WEIGHTS;demo.SOURCE=Path('D:/FPS_AI_Dataset/Apex_YOLO_DAY3')
    print('新候选：2.8秒补充数据联合n，第15轮best；左侧检测，右侧跟踪和绿色目标选择。',flush=True)
    print('保持旧联合n相同检测/跟踪参数；只做离线展示，不控制游戏。',flush=True)
    print('重点：E43只队友；E82约5秒背景误检；E320约1.4–1.7秒准心误检。七段是训练暴露后的回归诊断。',flush=True)
    demo.main(['--episodes','43','47','79','82','150','178','320','--split-view','--select-target',*sys.argv[1:]])
